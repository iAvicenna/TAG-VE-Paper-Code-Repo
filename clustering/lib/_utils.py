#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:31:27 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position


import numpy as np


def _sortlabels(labels):

  levels = sorted(list(set(labels)))
  order = [list(np.argsort([labels.index(x) for x in levels])).index(i)\
           for i in range(len(levels))]

  assignment = {x:y for x,y in zip(levels,order)}

  return np.array([assignment[x] for x in labels])


def split_data_by_missing_pattern(data):
  '''
  source: https://www.pymc.io/projects/examples/en/latest/howto/Missing_Data_Imputation.html
  We want to extract our the pattern of missing-ness in our dataset
  and save each sub-set of our data in a structure that can be used to
  feed into a log-likelihood function

  A pattern is whether the values in each column e.g. [True, True, True] or [True, True, False]
  '''

  grouped_patterns = []
  patterns = data.notnull().drop_duplicates().values
  observed = data.notnull()


  for p,pattern in enumerate(patterns):
    indices = [ind_label for ind_label,label in enumerate(observed.index)
         if all(observed.loc[label,:].values==pattern)]

    grouped_patterns.append([pattern, indices,
                             data.iloc[indices].dropna(axis=1)])

  return grouped_patterns


def _replace(x):
  x = x.replace("WT+BA.1","BA.1").replace("WT+BA.4/5","BA.4/5")

  return x


def _sort(x):

  return ["BA.5" in x, "BA.4" in x, "BA.2" in x, "BA.1" in x,
          "Delta" in x, x.count("WT")]


def combine_encounters(row):

  '''
  combines infection and vaccination encounters from a row in the meta table
  as infection + vaccination. It does some modifications with the function
  _replace.
  '''

  inf = row["infection"]
  vacc = row["vaccination"]

  if isinstance(inf, float) and np.isnan(inf):
    inf = ""
  if isinstance(vacc, float) and np.isnan(vacc):
    vacc = ""

  inf = inf.split(",")
  vacc = vacc.split(",")

  inf = [_replace(x) for x in inf if x.lower() not in ["","uninfected"]]
  vacc = [_replace(x) for x in vacc if x.lower() not in ["", "unvaccinated"]]

  if len(inf)>0:
    res =  '_'.join(sorted(inf, key=_sort)) + ' + '
  else:
    res = "None + "

  if len(vacc)>0:
    res = res + '_'.join(sorted(vacc, key=_sort))
  else:
    res = res + "None"

  return res
