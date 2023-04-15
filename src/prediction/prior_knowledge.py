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

import math
import numpy as np

def prior_knolwedge_categorized(judgment_var, n_bins = 12):
    """
    function of prior knowledge for categorized judgment_var
    :param judgment_var: input judgment_var - this judgment_var should already be bucketized. The value is in the range [0, n_bin)
    :param n_bins: number of bins
    :return: expected prob.
    """
    # The winning probably is linearly proportional to the inverse of judgment_var
    prob = 0.5 + (0.5) / (n_bins - 1) * judgment_var  # linear function for credit
    return prob
