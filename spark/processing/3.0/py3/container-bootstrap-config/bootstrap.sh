#!/bin/bash
set -x

python3 -m pip install pandas openpyxl==3.0.0 nltk NRCLex vaderSentiment bertopic==0.9.3 stanford_openie spacy wordninja emoji s3fs fsspec
python3 -m nltk.downloader popular
