#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:43:41 2024

@author: avicenna

"""

# pylint: disable=bad-indentation import-error

import pickle
import os

cdir = os.path.dirname(os.path.realpath(__file__))

import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

from run_landscapes import _get_model

cones = [1,2]
clusters = [0,1,2]


def _main(ncones_range=None, groups_range=None):

  if ncones_range is None:
    ncones_range = [1, 2]
  if groups_range is None:
    groups_range = [0, 1, 2]

  fig,ax = plt.subplots(1, 3, figsize=(15,5))
  nclusters = 3



  for i0 in groups_range:
    models = {}

    for ncones in cones:

      with open(f"{cdir}/outputs/group{i0}_ncones{ncones}","rb") as fp:
        idata, meta_data, _ = pickle.load(fp)

      model, _, _=\
       _get_model(nclusters=nclusters,  i0=i0, ncones=ncones,
                  **meta_data["model_args"])

      with model:
        pm.compute_log_likelihood(idata, extend_inferencedata=True)


      models[f"cluster{i0+1} cones{ncones}"] = idata

      idata_loo = az.loo(idata)
      print(idata_loo)

    if len(models)>1:
      loo_compare = az.compare(models)
      az.plot_compare(loo_compare, ax=ax[i0])
      ax[i0].set_title(f"Group {i0+1}")

    if i0>0:
      ax[i0].legend().remove()
      ax[i0].set_ylabel('')

  if len(cones)==2 and len(clusters)==3:
    fig.tight_layout()
    fig.savefig(f"{cdir}/plots/model_comparison.png")


if __name__ == "__main__":

  _main()
