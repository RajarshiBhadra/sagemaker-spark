# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import time

# Import local module to test spark-submit--py-files dependencies
import hello_py_spark_udfs as udfs
from pyspark.ml import Pipeline  # importing to test mllib DLLs like liblapack.so
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

from openie import StanfordOpenIE
import bertopic
import pandas
import openpyxl
import nltk
from nrclex import NRCLex
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import wordninja
import emoji
from s3fs.core import S3FileSystem
import fsspec
import pickle

import argparse
import os
import boto3
from pyspark.sql import SparkSession

from pyspark.sql.functions import lit
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, \
    TypeConverters,HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, size, collect_set
from pyspark.sql.types import ArrayType, StringType, FloatType,IntegerType,StructType,StructField, DoubleType,DateType
import pyspark.sql.functions as sf
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
import wordninja
import pickle
import nltk

if __name__ == "__main__":
    print("Hello World, this is PySpark!")

    parser = argparse.ArgumentParser(description="inputs and outputs")
    parser.add_argument("--input", type=str, help="path to input data")
    parser.add_argument("--output", required=False, type=str, help="path to output data")
    args = parser.parse_args()
    spark = SparkSession.builder.appName("SparkContainerTestApp").getOrCreate()
    print("Created spark context")
    sqlContext = SQLContext(spark.sparkContext)
    print("Created sql context")

#     # Load test data set
    inputPath = args.input
    salesDF = spark.read.json(inputPath)
    print("Loaded data")
    salesDF.printSchema()

    salesDF.createOrReplaceTempView("sales")
    topDF = spark.sql("SELECT date, sale FROM sales WHERE sale > 750")
    # Show the first 20 rows of the dataframe
    topDF.show()
    time.sleep(60)

#     # Calculate average sales by date
    averageSalesPerDay = salesDF.groupBy("date").avg().collect()
    print("Calculated data")

    outputPath = args.output

#     # Define a UDF that doubles an integer column
#     # The UDF function is imported from local module to test spark-submit--py-files dependencies
    double_udf_int = udf(udfs.double_x, IntegerType())

#     # Save transformed data set to disk
    salesDF.select("date", "sale", double_udf_int("sale").alias("sale_double")).write.json(outputPath)
    print("Saved data")

    print(nltk.__version__)

    data = [{"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            {"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            {"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            ]

    df = spark.createDataFrame(data)
    in_col  = "text"
    out_col ="words"
    t = StringType()
    df_processed = df.withColumn(out_col, udf(udfs.f, t)(in_col))
    df_processed.show()

    t = FloatType()
    out_col = "sentiment_score"
    in_col = "words"
    df_sent = df_processed.withColumn((out_col), udf(udfs.f_sent, t)(in_col))
    df_sent.show()

    t_score = ArrayType(IntegerType())
    t_label = ArrayType(StringType())
    out_col = ['emotion_score', 'emotion_label']
    in_col = "words"
    df_processed_emo = df_processed.withColumn((out_col[1]),
                                               udf(udfs.f1_emo, t_label)(in_col))
    output = df_processed_emo.withColumn((out_col[0]), udf(udfs.f_emo, t_score)(
        in_col))
    output.show()


