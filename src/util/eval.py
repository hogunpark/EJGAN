# Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import log2
from scipy.spatial import distance
import numpy as np
from scipy import spatial

def kl_divergence(p, q):
    """
    calculate the kl divergence
    p: observation (1-d list)
    q: theory (1-d list)
    """

    l_agg = []
    for i in range(len(p)):
        if q[i] == 0:
            l_agg.append(0)
        elif p[i] == 0:
            l_agg.append(0)
        else:
            l_agg.append(p[i] * log2(p[i]/q[i]))
    return sum(l_agg)

def js_distance(p, q, base =2):
    """
    calculate the js distance
    p: observation (1-d list)
    q: theory (1-d list)
    """
    dist = distance.jensenshannon(p, q, base)
    if np.isnan(dist):
        return 0.
    else:
        return 1-dist

def cos_distance(p,q):
    """
    calculate the cos distance
    p: observation (1-d list)
    q: theory (1-d list)
    """

    return 1 - spatial.distance.cosine(p, q)

def convert_str(a):
    """
    convert 1-d array to string for future p-value computation
    a: 1-d array (float)
    return string e.g. [1, 2, 3] -> "[1, 2", 3]"
    """
    s = ""
    for i in a:
        s += str(i) + ", "
    s = "[" + s[:-2] + "]"
    return s
