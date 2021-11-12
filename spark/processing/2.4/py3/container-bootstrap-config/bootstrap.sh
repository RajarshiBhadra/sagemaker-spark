#!/bin/bash
set -x

python3 -m pip install pandas openpyxl==3.0.0 nltk NRCLex vaderSentiment bertopic==0.9.3 stanford_openie spacy wordninja emoji s3fs fsspec spark-nlp==3.1.2 psutil
python3 -m nltk.downloader popular
