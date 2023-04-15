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

import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    """
    get a output from the sigmoid function
    :param x: input
    :return: output
    """
    return ((1 / (1 + np.exp(-1 * x))) - 0.5) * 2
def f2(x):
    """
        get a output from a linear function
        :param x: input
        :return: output
        """
    return x

def f3(x):
    """
        get a output from the sigmoid function
        :param x: input
        :return: output
        """
    return ((1 / (1 + np.exp(-1.5 * x))) - 0.5) * 2

def sigmoid(x, beta=1.0):
    """
        get a output from the sigmoid function
        :param x: input
        :return: output
        """
    return ((1 / (1 + np.exp(-1 * beta * x))) - 0.5) * 2


def f4(x):
    """
        get a output from the x^4
        :param x: input
        :return: output
        """
    return -1* np.power((x-1), 4)  + 1

