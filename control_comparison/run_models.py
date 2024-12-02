#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:16:59 2024

@author: avicenna
"""
# pylint: disable=bad-indentation, import-error, wrong-import-position

import sys
import pickle
import os
import pandas as pd
import pymc as pm

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{cdir}/lib/")
sys.path.append(f"{cdir}/../")

from models import gmt_bayesian_model
from common_utils import log, lower, upper

default_sample_args = {
  "draws":5000,
  "tune":2000,
  "chains":6,
  "cores":6,
  }


def _get_model(use_t, remove_outlier, prior_params):
  '''
  see main for variable descriptions

  this function loads the data and converts wide format
  (where antigens titres are in columns with values as measured titres)
  into flat format (where antigens are melted into rows and their values
  are recorded in the single column named titre)

  remove_outlier: To do some tests with and without RLID results
  which looks like an outlier

  '''

  table = pd.read_csv("../data/lab_data.csv", header=0, index_col=None)
  table = table[table.sample_id.isin(["NIBSC_21/338"])]

  if remove_outlier:
    table = table[~table.lab_code.isin(["RLID_AUE_EMRO"])]

  # process table so that it has flat format with an extra column
  # identifying antigen and value column giving titres
  flat_table =\
    pd.melt(table, id_vars=["lab_code","assay_type"],
            value_vars=["XBB.1.5", "Alpha", "Beta", "Delta", "BA.1", "BA.5"],
            var_name="antigen", value_name="titre")

  # add two columns corresponding to whether or not values are censored
  flat_table.loc[:,"lower"] =\
    [lower(x, y + '_' + z.lower()) for
     x,y,z in zip(flat_table.loc[:,"titre"], flat_table.loc[:,"lab_code"],
                  flat_table.loc[:,"assay_type"])]

  flat_table.loc[:,"upper"] =\
    [upper(x, y + '_' + z.lower()) for
     x,y,z in zip(flat_table.loc[:,"titre"], flat_table.loc[:,"lab_code"],
                  flat_table.loc[:,"assay_type"])]

  flat_table.loc[:,"titre"] = flat_table.loc[:,"titre"].apply(log).astype(float)

  flat_table.to_csv("./outputs/flat_table.csv", header=True, index=False)

  model, meta = gmt_bayesian_model(flat_table, use_t=use_t,
                                   prior_params=prior_params)

  return model, meta, flat_table



def main(use_t=True, remove_outlier=False, run=True):

  '''
  use_T: whether to use a student T for likelihood or normal
  remove_outlier: remove RLID data or not
  run: run model or load inference object
  '''
  prior_params = None

  if use_t:
    name_str = "T"
    sample_args = default_sample_args.copy()

  else:
    name_str = "N"
    sample_args = default_sample_args.copy()
    sample_args["default_accept"] = 0.9


  if remove_outlier:
    name_str += "_outrem"


  model, meta, flat_table = _get_model(use_t=use_t,
                                       remove_outlier=remove_outlier,
                                       prior_params=prior_params)

  if run:
    with model:
      idata = pm.sample(**default_sample_args)

    meta["sample_args"] = default_sample_args

    with open(f"{cdir}/outputs/fit_data_{name_str}","wb") as fp:
      pickle.dump([meta, idata, flat_table], fp)

  else:
    with open(f"{cdir}/outputs/fit_data_{name_str}","rb") as fp:
      meta, idata, flat_table = pickle.load(fp)


if __name__ == "__main__":

  main(remove_outlier=False) #student_T
  main(use_t=False, remove_outlier=False) #normal

  main(remove_outlier=True) #student_T
  main(use_t=False, remove_outlier=True) #normal
