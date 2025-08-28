# Use the official Python 3.9 image
FROM python:3.10

# Set the working directory to /code
WORKDIR /code

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /code
COPY ./pyproject.toml /code/pyproject.toml
COPY ./poetry.lock /code/poetry.lock
COPY ./index.md /code/index.md

# Install requirements.txt
RUN pip install poetry

ENV POETRY_VIRTUALENVS_CREATE=false

RUN poetry install

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

CMD ["poetry", "run", "streamlit", "run", "presidio_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]