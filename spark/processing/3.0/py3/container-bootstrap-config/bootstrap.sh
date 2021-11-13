#!/bin/bash
set -x

python3 -m pip install psutil
python3 -m pip install bertopic stanford_openie
python3 -m pip install pandas openpyxl==3.0.0 nltk NRCLex vaderSentiment 
python3 -m pip install spacy wordninja emoji s3fs fsspec spark-nlp==3.3.2 
python3 -m nltk.downloader popular
python3 -m pip uninstall numba
python3 -m pip install numba

