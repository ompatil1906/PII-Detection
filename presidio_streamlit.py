"""Streamlit app for Presidio."""
import logging
import os
from dotenv import load_dotenv

load_dotenv()
import traceback
import tempfile
import base64 # Added for PDF preview

# Configure logging to a file
logging.basicConfig(filename='streamlit_debug.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

import dotenv
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotated_text
from streamlit_tags import st_tags

from PIL import Image
import time
import boto3
import io
from PIL import Image


from openai_fake_data_generator import OpenAIParams
from presidio_helpers import (
    get_supported_entities,
    analyze,
    anonymize,
    annotate,
    create_fake_data,
    analyzer_engine,
)

st.set_page_config(
    page_title="Presidio demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")


allow_other_models = os.getenv("ALLOW_OTHER_MODELS", False)



# Add file upload options
def process_text_input(text):
    # Existing text processing logic
    # For now, simply return the text as is.
    # Future: Add pre-processing like cleaning, normalization if needed.
    return text


def get_text_from_image(image):
    client = boto3.client('textract', region_name=os.getenv('AWS_REGION', 'us-west-2'))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format if image.format else 'PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.detect_document_text(Document={'Bytes': img_byte_arr})

    text = ""
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + '\n'
    return text


def get_text_from_pdf(pdf_file):
    client = boto3.client('textract', region_name='us-west-2')
    # Upload PDF to S3 for Textract processing
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    s3_key = f'temp_pdfs/{os.path.basename(pdf_file.name)}'

    s3 = boto3.client('s3')
    s3.upload_fileobj(pdf_file, s3_bucket, s3_key)

    response = client.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket,
                'Name': s3_key
            }
        }
    )

    job_id = response['JobId']

    # Poll for job completion
    while True:
        job_status = client.get_document_text_detection(JobId=job_id)
        status = job_status['JobStatus']
        if status in ['SUCCEEDED', 'FAILED']:
            break
        time.sleep(5) # Wait 5 seconds before checking again

    if status == 'FAILED':
        return "Error processing PDF with Textract."

    text = ""
    pages = []
    next_token = None
    while True:
        if next_token:
            page_response = client.get_document_text_detection(JobId=job_id, NextToken=next_token)
        else:
            page_response = client.get_document_text_detection(JobId=job_id)

        for item in page_response['Blocks']:
            if item['BlockType'] == 'LINE':
                text += item['Text'] + '\n'
        
        next_token = page_response.get('NextToken', None)
        if not next_token:
            break

    # Clean up S3 object
    s3.delete_object(Bucket=s3_bucket, Key=s3_key)

    return text


# Sidebar
st.sidebar.header(
    """
PII De-Identification with [NEXYOM](https://www.nexyom.com)
"""
)


model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    Presidio supports multiple NER packages off-the-shelf, such as spaCy, Huggingface, Stanza and Flair,
    as well as service such as Azure Text Analytics PII.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    "spaCy/en_core_web_lg",
    "flair/ner-english-large",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "stanza/en",
    "Azure AI Language",
    "Other",
]
if not allow_other_models:
    model_list.pop()
# Select model
st_model = st.sidebar.selectbox(
    "NER model package",
    model_list,
    index=1,
    help=model_help_text,
)

# Extract model package.
st_model_package = st_model.split("/")[0]

# Remove package prefix (if needed)
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.sidebar.selectbox(
        "NER model OSS package", options=["spaCy", "stanza", "Flair", "HuggingFace"]
    )
    st_model = st.sidebar.text_input(f"NER model name", value="")

if st_model == "Azure AI Language":
    st_ta_key = st.sidebar.text_input(
        f"Azure AI Language key", value=os.getenv("TA_KEY", ""), type="password"
    )
    st_ta_endpoint = st.sidebar.text_input(
        f"Azure AI Language endpoint",
        value=os.getenv("TA_ENDPOINT", default=""),
        help="For more info: https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/personally-identifiable-information/overview",  # noqa: E501
    )


st.sidebar.warning("Note: Models might take some time to download. ")

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
logger.debug(f"analyzer_params: {analyzer_params}")

st_operator = st.sidebar.selectbox(
    "De-identification approach",
    ["redact", "replace", "synthesize", "highlight", "mask", "hash", "encrypt"],
    index=1,
    help="""
    Select which manipulation to the text is requested after PII has been identified.\n
    - Redact: Completely remove the PII text\n
    - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
    - Synthesize: Replace with fake values (requires an OpenAI key)\n
    - Highlight: Shows the original text with PII highlighted in colors\n
    - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
    - Hash: Replaces with the hash of the PII string\n
    - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
         """,
)
st_mask_char = "*"
st_number_of_chars = 15
st_encrypt_key = "WmZq4t7w!z%C&F)J"

open_ai_params = None

logger.debug(f"st_operator: {st_operator}")


def set_up_openai_synthesis():
    """Set up the OpenAI API key and model for text synthesis."""

    if os.getenv("OPENAI_TYPE", default="openai") == "Azure":
        openai_api_type = "azure"
        st_openai_api_base = st.sidebar.text_input(
            "Azure OpenAI base URL",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", default=""),
        )
        openai_key = os.getenv("AZURE_OPENAI_KEY", default="")
        st_deployment_id = st.sidebar.text_input(
            "Deployment name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", default="")
        )
        st_openai_version = st.sidebar.text_input(
            "OpenAI version",
            value=os.getenv("OPENAI_API_VERSION", default="2023-05-15"),
        )
    else:
        openai_api_type = "openai"
        st_openai_version = st_openai_api_base = None
        st_deployment_id = ""
        openai_key = os.getenv("OPENAI_KEY", default="")
    st_openai_key = st.sidebar.text_input(
        "OPENAI_KEY",
        value=openai_key,
        help="See https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key for more info.",
        type="password",
    )
    st_openai_model = st.sidebar.text_input(
        "OpenAI model for text synthesis",
        value=os.getenv("OPENAI_MODEL", default="gpt-3.5-turbo-instruct"),
        help="See more here: https://platform.openai.com/docs/models/",
    )
    return (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    )


if st_operator == "mask":
    st_number_of_chars = st.sidebar.number_input(
        "number of chars", value=st_number_of_chars, min_value=0, max_value=100
    )
    st_mask_char = st.sidebar.text_input(
        "Mask character", value=st_mask_char, max_chars=1
    )
elif st_operator == "encrypt":
    st_encrypt_key = st.sidebar.text_input("AES key", value=st_encrypt_key)
elif st_operator == "synthesize":
    (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    ) = set_up_openai_synthesis()

    open_ai_params = OpenAIParams(
        openai_key=st_openai_key,
        model=st_openai_model,
        api_base=st_openai_api_base,
        deployment_id=st_deployment_id,
        api_version=st_openai_version,
        api_type=openai_api_type,
    )

st_threshold = st.sidebar.slider(
    label="Acceptance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Define the threshold for accepting a detection as PII. See more here: ",
)

st_return_decision_process = st.sidebar.checkbox(
    "Add analysis explanations to findings",
    value=False,
    help="Add the decision process to the output table. "
    "More information can be found here: https://microsoft.github.io/presidio/analyzer/decision_process/",
)

# Allow and deny lists
st_deny_allow_expander = st.sidebar.expander(
    "Allowlists and denylists",
    expanded=False,
)

with st_deny_allow_expander:
    st_allow_list = st_tags(
        label="Add words to the allowlist", text="Enter word and press enter."
    )
    st.caption(
        "Allowlists contain words that are not considered PII, but are detected as such."
    )

    st_deny_list = st_tags(
        label="Add words to the denylist", text="Enter word and press enter."
    )
    st.caption(
        "Denylists contain words that are considered PII, but are not detected as such."
    )

# Input type selection and file upload
input_type = st.selectbox(
    "Select input type",
    options=["Text", "PDF", "Image"],
    index=0,
    help="Choose the type of input you want to analyze"
)

uploaded_file = None
if input_type == "PDF":
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload a PDF file for PII detection"
    )
elif input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image file",
        type=["png", "jpg", "jpeg"],
        help="Upload an image file for PII detection"
    )

# Main panel

analyzer_load_state = st.info("Starting Presidio analyzer...")

analyzer_load_state.empty()

# Read default text
with open("demo_text.txt") as f:
    default_demo_text = "".join(f.readlines())

# Initialize the text that will be displayed in the text area and used for analysis
current_input_text = default_demo_text



# Create two columns for before and after
col1, col2 = st.columns(2)

# Before:
col1.subheader("Input")

if uploaded_file is not None:
    logging.debug(f"Uploaded file detected. Input type: {input_type}")
    if input_type == "PDF":
            logging.debug("Entered PDF processing block.") # Debugging line
            try:
                # Get PDF content for preview before any other operations
                pdf_content_for_preview = uploaded_file.getvalue()
                logging.debug(f"PDF content length: {len(pdf_content_for_preview)} bytes") # Debugging line
                # Reset the file pointer to the beginning after reading for preview
                uploaded_file.seek(0)

                text = get_text_from_pdf(uploaded_file)
                current_input_text = process_text_input(text)
                output_path = os.path.join("/Users/ompatil/Om/Projects/Redact/presidio_demo", f"extracted_{uploaded_file.name}.txt")
                with open(output_path, "w") as output_file:
                    output_file.write(current_input_text)
                st.success(f"Text extracted and saved to: {output_path}")

                # Display PDF preview
                base64_pdf = base64.b64encode(pdf_content_for_preview).decode('utf-8')
                logging.debug(f"Base64 PDF string length: {len(base64_pdf)} characters") # Debugging line
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                col1.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                current_input_text = default_demo_text
    elif input_type == "Image":
            try:
                img = Image.open(uploaded_file)
                
                col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                text = get_text_from_image(img)
                current_input_text = process_text_input(text)
                output_path = os.path.join("/Users/ompatil/Om/Projects/Redact/presidio_demo", f"extracted_{uploaded_file.name}.txt")
                with open(output_path, "w") as output_file:
                    output_file.write(current_input_text)
                st.success(f"Text extracted and saved to: {output_path}")
            except Exception as e:
                st.error(f"Error processing image: {e}")
                current_input_text = default_demo_text

if input_type == "PDF":
    st_text = col2.text_area(
        label="Extracted text from PDF", value=current_input_text, height=400, key="pdf_text_input", disabled=True
    )
elif input_type == "Image":
    st_text = col1.text_area(
        label="Extracted text from Image", value=current_input_text, height=400, key="image_text_input", disabled=True
    )
elif input_type == "Text":
    st_text = col1.text_area(
        label="Enter text", value=current_input_text, height=400, key="text_input"
    )
else:
    st_text = col1.text_area(
        label="Enter text", value=current_input_text, height=400, key="default_text_input"
    )

try:
    # Choose entities
    st_entities_expander = st.sidebar.expander("Choose entities to look for")
    st_entities = st_entities_expander.multiselect(
        label="Which entities to look for?",
        options=get_supported_entities(*analyzer_params),
        default=list(get_supported_entities(*analyzer_params)),
        help="Limit the list of PII entities detected. "
        "This list is dynamic and based on the NER model and registered recognizers. "
        "More information can be found here: https://microsoft.github.io/presidio/analyzer/adding_recognizers/",
    )

    # Before
    analyzer_load_state = st.info("Starting Presidio analyzer...")
    analyzer = analyzer_engine(*analyzer_params)
    analyzer_load_state.empty()

    st_analyze_results = analyze(
        *analyzer_params,
        text=st_text,
        entities=st_entities,
        language="en",
        score_threshold=st_threshold,
        return_decision_process=st_return_decision_process,
        allow_list=st_allow_list,
        deny_list=st_deny_list,
    )

    # After
    if st_operator not in ("highlight", "synthesize"):
        with col2:
            st.subheader(f"Output")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="De-identified", value=st_anonymize_results.text, height=400
            )
    elif st_operator == "synthesize":
        with col2:
            st.subheader(f"OpenAI Generated output")
            fake_data = create_fake_data(
                st_text,
                st_analyze_results,
                open_ai_params,
            )
            st.text_area(label="Synthetic data", value=fake_data, height=400)
    else:
        st.subheader("Highlighted")
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        # annotated_tokens
        annotated_text(*annotated_tokens)

    # table result
    st.subheader(
        "Findings"
        if not st_return_decision_process
        else "Findings with decision factors"
    )
    if st_analyze_results:
        df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
        df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

        df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
            {
                "entity_type": "Entity type",
                "text": "Text",
                "start": "Start",
                "end": "End",
                "score": "Confidence",
            },
            axis=1,
        )
        df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
        if st_return_decision_process:
            analysis_explanation_df = pd.DataFrame.from_records(
                [r.analysis_explanation.to_dict() for r in st_analyze_results]
            )
            df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
    else:
        st.text("No findings")

except Exception as e:
    print(e)
    traceback.print_exc()
    st.error(e)

components.html(
    """
    <script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "h7f8bp42n8");
    </script>
    """
)
