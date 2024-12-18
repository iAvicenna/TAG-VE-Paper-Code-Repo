#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:11:26 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import sys
from collections import Counter
import os

cdir = os.path.dirname(os.path.realpath(__file__))

import pandas as pd
import numpy as np

sys.path.append(f"{cdir}/../clustering/lib/")
from _utils import combine_encounters


def _sort(x):

  return ["BA.5" in x, "BA.4" in x, "BA.2" in x, "BA.1" in x,
          "Delta" in x, x.count("WT")]


def _replace(x):
  x = x.replace("WT+BA.1","BA.1").replace("WT+BA.4/5","BA.4/5")

  return x


table = pd.read_csv(f"{cdir}/../data/lab_data.csv", header=0)
table = table[~table.sample_id.isin(["NIBSC_21/338"])]
res = Counter(table.T.apply(combine_encounters))

N = np.sum(list(res.values()))
counter = 0
for key in res:

  inf = key.split(' + ')[0]
  vacc = key.split(' + ')[1]

  if inf == "None" or inf=="unknown" and "WT" in vacc:
    counter += res[key]
  else:
    print(key)

table=\
pd.DataFrame(res.values(), index=res.keys(), columns=["Count"])
table.index.name = "Infection + Vaccination"
table = table.sort_values("Count").iloc[::-1,:]
table.to_csv(f"{cdir}/outputs/categories_table.csv")
