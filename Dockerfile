##!/bin/bash
FROM python:3.8
MAINTAINER Pritam Sinha

# Astra ID
ENV ASTRA_DB_ID='a7631837-1639-436e-98fe-b1095213fc04'
ENV ASTRA_DB_REGION='asia-south1'
ENV ASTRA_DB_APPLICATION_TOKEN='AstraCS:LtfxrNnuoRoZvcFCBCQLvZAI:0fcdaf8c4e542abb17c4273c591e319b5fd64022b844619e5009be1f590703c2'
ENV ASTRA_DB_KEYSPACE='news_category'
ENV ASTRA_DB_COLLECTION_1='news_test'
ENV ASTRA_DB_COLLECTION_2='news_test_labels'

# Tensorflow Log Level
ENV TF_CPP_MIN_LOG_LEVEL='3'

COPY . /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data')" ]
EXPOSE 5000
CMD [ "python", "main.py", "flask", "run", "-h", "0.0.0.0", "-p", "5000" ]