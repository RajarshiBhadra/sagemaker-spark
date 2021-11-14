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




