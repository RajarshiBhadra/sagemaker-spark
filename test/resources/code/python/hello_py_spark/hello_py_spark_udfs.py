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
def double_x(x):
    return x + x

import re
import wordninja
from nrclex import NRCLex
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def f(s):
    s = re.sub(
        r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{"
        r"4}|\d{3}[-\.\s]??\d{4})",
        '', s)
    s = re.sub(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:["
        r"^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\(["
        r"^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))",
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
def f_sent(s):
    analyser = SentimentIntensityAnalyzer()
    getscore = analyser.polarity_scores(s)
    getscore = getscore['compound']
    print("Performing Sentiment Analysis")
    return getscore
def f1_emo(s):
    category = ['fear', 'anger', 'negative', 'disgust', 'sadness',
                'anticipation', 'trust', 'surprise', 'positive', 'joy']
    return category
def f_emo(s):
    lower_s = s.lower()
    out = emotion_fit(lower_s)
    return out


def ant_finder(emotion):
        """
        Antonym finder - Finds antonym for any given word
        :param emotion: Word to find Antonym for
        :return: Antonym of the emotion/word
        """
        # nltk.data.path.append("/root/nltk_data")
        # nltk.data.path.append('/libs/nltk_data/')

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
        Method 2: if method 1 doesn't work, use opps dict to guess opposite
        emotion
        :return:
        """
        negatives = "aren't, can't, couldn't, daren't, didn't, doesn't, " \
                    "don't," \
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

                    # Method 1: find opposite of emotive word and rerun NRC
                    # on it
                    print("Step 2")
                    ant = ant_finder(w)
                    if (ant):
                        temp = NRCLex(ant[0])
                        if (temp.affect_dict):
                            new_dict[ant[0]] = temp.affect_dict[ant[0]]
                            break

                    # Method 2: if method 1 doesn't work, use opps dict to
                    # guess opposite emotion
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
        import os
        os.listdir(".")
        print("1st Check")
        os.environ['HOME'] = '/home/hadoop'
        nrclex_out = NRCLex(str(clean_cell))
        print("1st Check Completed")
        out = negative_emotion_handler(nrclex_out)

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






