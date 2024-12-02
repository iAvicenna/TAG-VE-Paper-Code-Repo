#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:12:50 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import sys
import os
import pickle
import pandas as pd
import numpy as np

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/lib/")

from clustering import kmeans_missing

sys.path.append(f"{cdir}/../")

from common_utils import log, lower, upper, antigens, min_non_thresholded


def _compare(s0, s1, e0, e1):

  '''
  win if if score is bigger and error is not horrible or score is much bigger
  else one with biggest score wins
  '''

  if (s1>=s0 and e1<=1.1*e0) or s1>=1.4*s0:
    return 1

  return 0


def _subset_table(table, data_to_take):

  assay_lab = [(y,x) for x,y in zip(table.loc[:,"assay_type"].values,
                                    table.loc[:,"lab_code"].values)]

  I = [ind for ind,val in enumerate(assay_lab) if val in data_to_take]
  return table.iloc[I,:]


def _process_titre_table(titre_table, lab_assay_codes):

  titre_table = titre_table.T
  lab_assay_codes = np.tile(np.array(lab_assay_codes)[None,:],
                            (titre_table.shape[0],1))
  lower_vals = [lower(x,y) for x,y in zip(titre_table.values.flatten(),
                                         lab_assay_codes.flatten())]

  upper_vals = [upper(x,y) for x,y in zip(titre_table.values.flatten(),
                                         lab_assay_codes.flatten())]

  lower_table = pd.DataFrame(np.reshape(lower_vals, titre_table.shape),
                             index=titre_table.index,
                             columns=titre_table.columns)
  upper_table = pd.DataFrame(np.reshape(upper_vals, titre_table.shape),
                             index=titre_table.index,
                             columns=titre_table.columns)
  titre_table = titre_table.map(log)

  I0 = (lower_table.values != titre_table) & ~(np.isnan(titre_table))
  I1 = (upper_table != titre_table) & ~(np.isnan(titre_table))
  I = np.where((I0.sum(axis=0)>=min_non_thresholded) & (I1.sum(axis=0)>=min_non_thresholded))[0]

  titre_table = titre_table.iloc[:,I]
  lower_table = lower_table.iloc[:,I]
  upper_table = upper_table.iloc[:,I]


  return titre_table.T, lower_table.T, upper_table.T


def _combine_outputs(outputs, nclusters):

  I = [0 for _ in range(nclusters)] #best run index for each cluster

  for i0 in range(1,nclusters+1):
    for i1,output1 in enumerate(outputs):
      e0,s0 = outputs[I[i0-1]][i0]["err"], outputs[I[i0-1]][i0]["score"]
      e1,s1 = output1[i0]["err"], output1[i0]["score"]

      if _compare(s0, s1, e0, e1) == 1:
        I[i0-1] = i1

  combined_output = {}
  for i in range(1,nclusters+1):
    i0 = I[i-1]
    combined_output[i] = outputs[i0][i]

  with open(f"{cdir}/outputs/kmeans/combined","wb") as fp:
    pickle.dump(combined_output, fp)


def _run_kmeans(table, nclusters, name):

  lab_assay_codes = [f"{x}_{y.lower()}" for x,y in zip(table.loc[:,"lab_code"],
                             table.loc[:,"assay_type"])]

  titre_table = table.loc[:,["serum_long"]+antigens]
  titre_table.set_index("serum_long", inplace=True)


  titre_table, lower_table, upper_table=\
    _process_titre_table(titre_table, lab_assay_codes)

  with open(f"{cdir}/outputs/tables","wb") as fp:
    pickle.dump([titre_table, lower_table, upper_table], fp)


  output = kmeans_missing(titre_table.values, nclusters, max_iter=500,
                          normalize=True)

  with open(f"{cdir}/outputs/kmeans/{name}","wb") as fp:
    pickle.dump(output, fp)



def main(run=True, combine=True, nclusters=9):

  '''
  clusters the datasets given by data_to_take below into
  1,...,nclusters clusters using kmeans.

  if combine=True, it combines the results into a best run per each number of
  clusters based on the function _compare

  if run=False and combine=True, it just combines existing output if any
  '''

  data_to_take = [ ('EVC_US_PAHO', 'FRNT'),  ('SPH_HK_WPRO', 'PRNT'), ('MUI_AT_EURO', 'FRNT'),
                  ('NICD_SA_AFRO', 'PNT'), ('VIDRL_AU_WPRO', 'MN'), ('FIVI_CH_EURO','MN'),
                  ('UHG_CH_EURO', 'FRNT'), ('AHRI_ZA_AFRO','FRNT'),
                  ('FIOCRUZ_BS_PAHO','PRNT')]


  table = pd.read_csv("../data/lab_data.csv", header=0, index_col=None)
  table = table[~table.serum_long.isin(["NIBSC_21/338"])]


  table = _subset_table(table, data_to_take)

  table.to_csv(f"{cdir}/outputs/subsetted_table.csv", header=True,
               index=False)
  outputs = []

  for run_id in range(10):
    print(f"run:{run_id}")
    name = f"run{run_id}"

    if run:
      _run_kmeans(table, nclusters, name)

    if combine:
      with open(f"{cdir}/outputs/kmeans/{name}","rb") as fp:
        outputs.append(pickle.load(fp))

  if combine:
    _combine_outputs(outputs, nclusters)


if __name__ == "__main__":

  main(True, True)
