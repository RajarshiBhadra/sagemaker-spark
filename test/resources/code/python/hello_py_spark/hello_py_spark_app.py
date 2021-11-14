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

if __name__ == "__main__":
    print("Hello World, this is PySpark!")

#     parser = argparse.ArgumentParser(description="inputs and outputs")
#     parser.add_argument("--input", type=str, help="path to input data")
#     parser.add_argument("--output", required=False, type=str, help="path to output data")
#     args = parser.parse_args()
    spark = SparkSession.builder.appName("SparkContainerTestApp").getOrCreate()
    print("Created spark context")
    sqlContext = SQLContext(spark.sparkContext)
    print("Created sql context")

#     # Load test data set
#     inputPath = args.input
#     salesDF = spark.read.json(inputPath)
#     print("Loaded data")
#     salesDF.printSchema()

#     salesDF.createOrReplaceTempView("sales")
#     topDF = spark.sql("SELECT date, sale FROM sales WHERE sale > 750")
#     # Show the first 20 rows of the dataframe
#     topDF.show()
#     time.sleep(60)

#     # Calculate average sales by date
#     averageSalesPerDay = salesDF.groupBy("date").avg().collect()
#     print("Calculated data")

#     outputPath = args.output

#     # Define a UDF that doubles an integer column
#     # The UDF function is imported from local module to test spark-submit--py-files dependencies
#     double_udf_int = udf(udfs.double_x, IntegerType())

#     # Save transformed data set to disk
#     salesDF.select("date", "sale", double_udf_int("sale").alias("sale_double")).write.json(outputPath)
#     print("Saved data")
    
    #Test Code
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

    data = [{"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            {"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            {"text": 'This is a great book', "primary_key": 1, "batch_id": 1},
            ]

    df = spark.createDataFrame(data)

    class TextPreprocessor(
        Transformer, HasInputCol, HasOutputCol,
        DefaultParamsReadable, DefaultParamsWritable):

        @keyword_only
        def __init__(self, inputCol=None, outputCol=None, stopwords=None):
            super(TextPreprocessor, self).__init__()
            kwargs = self._input_kwargs
            self.setParams(**kwargs)

        @keyword_only
        def setParams(self, inputCol=None, outputCol=None, stopwords=None):
            kwargs = self._input_kwargs
            return self._set(**kwargs)

        def setInputCol(self, value):
            return self._set(inputCol=value)

        def setOutputCol(self, value):
            return self._set(outputCol=value)

        def _transform(self, dataset):
            help("nltk")
            def f(s):

                s = re.sub(
                    r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})",
                    '', s)
                s = re.sub(
                    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
                    '', s)
                s = re.sub(r"\w+@\w+.[\w+]{2,4}$", '', s)

                s = s.replace(":", " ").replace(";", " ").replace("-", " ")

                s = s.replace('*', '').replace(',', ' ')

                s = s.replace("(", " ").replace(')', ' ')

                s = re.sub('\.\.+', '. ', s)

                s = s.replace('  ', ' ')

                for i in s.split():
                    if '#' in i:
                        s = s.replace(i, ' '.join(wordninja.split(i)))
                return s

            t = StringType()
            out_col = self.getOutputCol()
            in_col = dataset[self.getInputCol()]
            print("Completed Text Preprocessor")
            return dataset.withColumn(out_col, udf(f, t)(in_col))


    class SentimentAnalyzer(
        Transformer, HasInputCol, HasOutputCol,
        DefaultParamsReadable, DefaultParamsWritable):

        @keyword_only
        def __init__(self, inputCol=None, outputCols=None, stopwords=None):
            super(SentimentAnalyzer, self).__init__()
            kwargs = self._input_kwargs
            self.setParams(**kwargs)

        @keyword_only
        def setParams(self, inputCol=None, outputCol=None, stopwords=None):
            kwargs = self._input_kwargs
            return self._set(**kwargs)

        def setInputCol(self, value):
            return self._set(inputCol=value)

        def setOutputCol(self, value):
            return self._set(outputCol=value)

        def _transform(self, dataset):
            def f(s):
                analyser = SentimentIntensityAnalyzer()
                getscore = analyser.polarity_scores(s)
                getscore = getscore['compound']
                print("Performing Sentiment Analysis")
                return getscore

            t = FloatType()
            out_col = self.getOutputCol()
            in_col = dataset[self.getInputCol()]
            return dataset.withColumn((out_col), udf(f, t)(in_col))

    def ant_finder(emotion):
        """
        Antonym finder - Finds antonym for any given word
        :param emotion: Word to find Antonym for
        :return: Antonym of the emotion/word
        """
        # nltk.data.path.append("/root/nltk_data")
        # nltk.data.path.append('/libs/nltk_data/')
        import nltk
        help("nltk")
        from nltk.corpus import wordnet
        ant = []
        for ss in wordnet.synsets(emotion):
            for lemma in ss.lemmas():
                if lemma.antonyms():
                    ant.append(lemma.antonyms()[0].name())
        return ant
    def negative_emotion_handler(obj):
        """
        Checks if any of the words negate the emotive word in question
        Method 1: find opposite of emotive word and rerun NRC on it
        Method 2: if method 1 doesn't work, use opps dict to guess opposite emotion
        :return:
        """
        negatives = "aren't, can't, couldn't, daren't, didn't, doesn't, don't," \
                    " hasn't, haven't, hadn't, isn't, mayn't, mightn't, " \
                    "mustn't, needn't, oughtn't, shan't, shouldn't, wasn't," \
                    " weren't, won't, wouldn't"
        negatives = negatives.replace("'", "").split(", ") + negatives.replace(
            "'", " ").split(", ") + ["not"]
        opps = {
            "fear": "trust",  #
            "anger": 'surprise',  #
            "anticipation": "disgust",  #
            "trust": 'fear',  #
            "surprise": 'anger',  #
            "positive": 'negative',  #
            "negative": 'positive',  #
            "sadness": 'joy',  #
            "disgust": 'anticipation',  #
            "joy": "sadness"  #
        }

        new_dict = {}
        for w in obj.affect_dict.keys():
            # Get three words before identified emotive word
            pos = obj.words.index(w)
            if pos < 3:
                check = obj.words[0:pos]
            else:
                check = obj.words[pos - 3:pos]

            bl = 1
            # Check if any of the words negate the emotive word in question
            for n in negatives:
                if n in check:
                    bl = 0

                    # Method 1: find opposite of emotive word and rerun NRC on it
                    ant = ant_finder(w)
                    if (ant):
                        temp = NRCLex(ant[0])
                        if (temp.affect_dict):
                            new_dict[ant[0]] = temp.affect_dict[ant[0]]
                            break

                    # Method 2: if method 1 doesn't work, use opps dict to guess opposite emotion
                    opp_emotions = []
                    for em in obj.affect_dict[w]:
                        opp_emotions.append(opps[em])
                    new_dict['not ' + w] = opp_emotions
                    break
            if (bl):
                new_dict[w] = obj.affect_dict[w]
        return (new_dict)
    def emotion_fit(clean_cell):
        """
        Organize the data in desired format with Negative and Positive Keys
        :param clean_cell: Input to run Emotion analysis on
        :return: row with all emotions and their score
        """
        pos_emotions = ['anticipation', 'trust', 'surprise', 'positive', 'joy']
        all_emotions = ['fear', 'anger', 'negative', 'disgust', 'sadness',
                        'anticipation', 'trust', 'surprise', 'positive', 'joy']

        list_of_emotions = []
        pos, neg = '', ''
        help("nltk")
        import os
        os.listdir(".")
        out = negative_emotion_handler(NRCLex(str(clean_cell)))

        temp = []
        for key in out.keys():
            temp = temp + out[key]

            if out[key][0] in pos_emotions:
                pos = pos + key + ","
            else:
                neg = neg + key + ","

        # row = [pos.strip(","), neg.strip(",")]
        row = []
        for emotion in all_emotions:
            row.append(temp.count(emotion))

        list_of_emotions.append(row)

        return list_of_emotions[0]

    class EmotionAnalyzer(
        Transformer, HasInputCol, HasOutputCols,
        DefaultParamsReadable, DefaultParamsWritable):

        @keyword_only
        def __init__(self, inputCol=None, outputCol=None, stopwords=None):
            super(EmotionAnalyzer, self).__init__()
            kwargs = self._input_kwargs
            self.setParams(**kwargs)

        @keyword_only
        def setParams(self, inputCol=None, outputCol=None, stopwords=None):
            kwargs = self._input_kwargs
            return self._set(**kwargs)

        def setInputCol(self, value):
            return self._set(inputCol=value)

        def setOutputCols(self, value):
            """
            Sets the value of :py:attr:`outputCol`.
            """
            return self._set(outputCols=value)

        def _transform(self, dataset):
            def f(s):
                lower_s = s.lower()
                out = emotion_fit(lower_s)
                print("performing Emotion Analysis")
                return out

            def f1(s):
                category = ['fear', 'anger', 'negative', 'disgust', 'sadness',
                            'anticipation', 'trust', 'surprise', 'positive', 'joy']
                return category


            t_score = ArrayType(IntegerType())
            t_label = ArrayType(StringType())
            out_col = self.getOutputCols()
            in_col = dataset[self.getInputCol()]
            print("Starting Emotion Analysis")
            dataset = dataset.withColumn((out_col[1]), udf(f1, t_label)(in_col))
            print("Completed Emotion Analysis f1")
            output = dataset.withColumn((out_col[0]), udf(f, t_score)(in_col))
            print("Completed Emotion Analysis f")
            return output

    text_preprocessor = TextPreprocessor(
            inputCol="text", outputCol="words")

    emotion_analysis = EmotionAnalyzer().setInputCol("words").setOutputCols(
            ['emotion_score', 'emotion_label'])
    sentiment_analysis = SentimentAnalyzer().setInputCol("words").setOutputCol(
        'sentiment_score')

    ner_prediction_pipeline = Pipeline(
            stages=[
                text_preprocessor,
                sentiment_analysis,
                emotion_analysis
            ])

    columns = StructType([StructField('text',
                                      StringType(), True),
                          StructField('primary_key',
                                      StringType(), True),
                          StructField('batch_id',
                                      StringType(), True)])

    empty_data = spark.createDataFrame(data=[],
                               schema=columns)

    prediction_model = ner_prediction_pipeline.fit(empty_data)
    predictions = prediction_model.transform(df)
    predictions.show(10)
    
    
    
