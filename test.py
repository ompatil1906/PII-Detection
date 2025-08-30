import os

# This example requires environment variables named "LANGUAGE_KEY" and "LANGUAGE_ENDPOINT"
language_key = os.environ.get('LANGUAGE_KEY')
language_endpoint = os.environ.get('LANGUAGE_ENDPOINT')

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Authenticate the client using your key and endpoint 
def authenticate_client():
    ta_credential = AzureKeyCredential(language_key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=language_endpoint, 
            credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

# Example method for detecting sensitive information (PII) from text 
def pii_recognition_example(client):
    document = """GST IN - 09AAPCS9575E1ZL, Kanpur Nagar - 209304
Invoice Number
IU167FT1J169379189
Date16 December, 2022
AWB Number
2827719809855
Payment Method
Credit Card
SHIP FROM
Piyush Tyagi
C-5, Sanjay Vihar , Avas Vikas , Meerut Road , Hapur,
Hapur, uttar pradesh, 245101
SHIP TO
Harsh Singh Rajput
293, Ground Floor,Urban State , Sector 7, Gurgaon,
haryana, 122001
ITEM
ITEM VALUE
SHIPPING CHARGES
Consumables/FMCG - Non-perishable food items
₹500.00
₹138.98
Delhivery Protect
₹0.00
SGST @ 9%
₹12.51
CGST @ 9%
₹12.51
Discount
-₹0.00
Invoice Total
₹164.00
Paid
-₹164.00
Amount Due
₹0.00
Delhivery LTD - Plot 5, Sector 44, Gurgaon 122002
This is a system generated invoice and does not require additional authentication"""
    response = client.recognize_pii_entities([document], language="en")
    result = [doc for doc in response if not doc.is_error]
    for doc in result:
        print("Redacted Text: {}".format(doc.redacted_text))
        for entity in doc.entities:
            print("Entity: {}".format(entity.text))
            print("\tCategory: {}".format(entity.category))
            print("\tConfidence Score: {}".format(entity.confidence_score))
            print("\tOffset: {}".format(entity.offset))
            print("\tLength: {}".format(entity.length))
pii_recognition_example(client)