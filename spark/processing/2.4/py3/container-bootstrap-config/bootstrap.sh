#!/bin/bash
set -x

python3 -m pip install psutil pandas openpyxl==3.0.0 nltk NRCLex vaderSentiment
python3 -m pip install bertopic==0.9.3 stanford_openie spacy wordninja emoji s3fs fsspec spark-nlp
python3 -m nltk.downloader popular
