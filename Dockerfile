# Select a base image
# https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
# Use the miminal-notebook as base container
ARG BASE_CONTAINER=jupyter/minimal-notebook
# ARG BASE_CONTAINER=jupyter/r-notebook

FROM $BASE_CONTAINER

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install all Python dependencies
RUN python3 -m pip install -r requirements.txt
