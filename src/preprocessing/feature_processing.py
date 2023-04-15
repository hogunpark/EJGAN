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

import numpy as np
import pandas as pd
import collections
from math import log10, floor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from numpy import genfromtxt

def round_decimal(x, dec = 2):
    """
    to round up given the number of decimal points
    :param x: input
    :param dec: number of decimal points
    :return: output
    """
    return np.around(x, decimals=dec)


def getBinID(value, n_bins = 10, min = 0.0, max = 1.0):
    """
    To return a categoried ID
    :param value: input values
    :param n_bins: number of bins
    :param min: the min value for all bins
    :param max: the max vlue for all bins
    :return: new categorized values (possible bins(total: n_bins + 2) : 0 - n_bins + 1)
    """

    unit_size = float((max - min) / n_bins) # size of unit of values
    l_bin = []

    # initialize the bin boundaries for numpy.digitize
    for i in range(n_bins):
        l_bin.append(min + unit_size * i)
    l_bin.append(max)

    return np.digitize(value, l_bin)

def split_data(n_data, seed = 10, ratio = 0.8 ):
    """
    to divide data into training and testing data
    :param n_data: the number of data (we assume that data ID is between 0 and n_data - 1)
    :param seed: random number seed
    :param ratio: ratio of training data
    :return: ids of training data and testing data
    """

    l_ID = np.arange(n_data)
    np.random.seed(seed)
    np.random.shuffle(l_ID)

    id_train = l_ID[:int(len(l_ID) * ratio)]
    id_test = l_ID[int(len(l_ID) * ratio):]

    return id_train, id_test

def categorization_feature(data, n_bins = 12):
    """
    to make one hot vectors from categorized inputs (e.g. [0, 2] -> [[1,0,0], [0,0,1]] )
    :param data: [N X 1] input
    :param n_bins: number of bins
    :return: one hot vectors
    """

    length = data.shape[0] # number of rows
    cat_data = np.zeros((length, n_bins))

    for i in range(length):
        cat_data[i, data[i]] = 1.0

    return cat_data


def adjust_judgment(all_judgment, all_label, id_minor, distance):
    """
    to adjust judgment using distance (here, add distatnce)
    :param all_judgment: all values about instances [N, 1]
    :param all_label: all labels [N, 1]
    :param id_minor: all ids in minor cluster [M, 1] where N+M = # of all instances
    :param distance: distance
    :return: adjusted_judgment
    """
    new_judgment = np.array(all_judgment)

    for i in range(len(all_judgment)):
        if i in id_minor and all_label[i] == 1:
            new_judgment[i] = all_judgment[i] + distance
            if new_judgment[i] > 1.0: new_judgment[i] = 1.0
            elif new_judgment[i] < 0.0: new_judgment[i] = 0.0

    return new_judgment


def get_distance(id_major, id_minor, all_judgment):
    """
    to get the difference between averaged minor instances from averaged major instances
    :param id_major: all ids in major cluster [N, 1] where N+M = # of all instances
    :param id_minor: all ids in minor cluster [M, 1] where N+M = # of all instances
    :param all_judgment: all values about instances [N+M, 1]
    :return: distance (float)
    """
    l_major = []
    l_minor = []

    for i in range(len(all_judgment)):
        if i in id_major:
            l_major.append(all_judgment[i])
        elif i in id_minor:
            l_minor.append(all_judgment[i])

    center_major = np.mean(l_major)
    center_minor = np.mean(l_minor)

    distance = abs(center_major - center_minor)

    if center_minor >= center_major:
        return distance
    else:
        return -1 * distance


def split_data_byClustering(X, ratio = 0.1, option = 1):
    """
    to select two clusters and return their ids
    :param X: attribute vectors [N, # of attributes] where N is the number of instances
    :param ratio: percentage of outlier for option 1 and 2
    :param opton: 0-kmean, 1-isolation forest, 2-localoutlierfactor
    :return: larger cluster ids [M] , smaller cluster ids [P]  where M+P = N
    """
    # option0 : kmean
    if option == 0:
        kmeans = KMeans(n_clusters=2, random_state=10).fit(X)
        labels = kmeans.labels_

        set1 = np.where(labels == 1)[0]
        set2 = np.where(labels == 0)[0]

    # option1: isolation forest
    elif option == 1:
        detector = IsolationForest(contamination=ratio,
                                             random_state=42)

        fitted_detector = detector.fit(X)
        labels = fitted_detector.predict(X)


        set1 = np.where(labels == 1)[0]
        set2 = np.where(labels == -1)[0]

    elif option == 2:
        detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=ratio)

        labels = detector.fit_predict(X)
        set1 = np.where(labels == 1)[0]
        set2 = np.where(labels == -1)[0]


    if len(set1) >= len(set2): # major first
        return set1, set2
    else:
        return set2, set1



def getcreditdata(inputfile, n_bins = 12, op_value = 1):
    dataset = genfromtxt(inputfile, delimiter=',')

    features = dataset[:, :-2]  # we will use 38
    df_label = dataset[:, -1]

    all_judgment = features[:, 38]
    df_feature = np.hstack([features[:, :38], features[:, 39:]])

    all_feature = np.array(df_feature)
    all_label = np.array(df_label, dtype=int)
    all_judgment = np.array(all_judgment)


    all_judgment = np.reshape(all_judgment, (len(all_judgment)))
    n_classes = len(set(list(all_label)))

    if op_value == 1:
        for i in range(all_judgment.shape[0]):
            if all_judgment[i] == 1:
                all_judgment[i] = n_bins-1

    return df_feature, df_label, all_judgment


