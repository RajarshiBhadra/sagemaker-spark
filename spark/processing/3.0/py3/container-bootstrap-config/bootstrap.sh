#!/bin/bash
set -x

python3 -m pip install bertopic==0.9.3 stanford_openie
python3 -m pip install psutil pandas openpyxl==3.0.0 nltk NRCLex vaderSentiment 
python3 -m pip install spacy wordninja emoji s3fs fsspec spark-nlp==3.3.2 
python3 -m nltk.downloader popular

