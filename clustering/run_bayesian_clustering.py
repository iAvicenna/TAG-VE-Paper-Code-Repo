#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:34:16 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import os
import sys
import pickle
from datetime import datetime

cdir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pymc as pm
import arviz as az

sys.path.append(f"{cdir}/lib/")
from clustering import censored_titre_clustering_model

sys.path.append(f"{cdir}/../")
from common_utils import antigens


default_sample_args = dict({"draws":2000, "chains":4, "tune":3000, "cores":4,
                            "init":"advi+adapt_diag", "target_accept":0.9})


def run_bayesian_clustering(nclusters_range, sr_bias_sd=0.5, centre_sd=1,
                            outlier_sd_factor=4, kmeans_path=None,
                            sample_args=None):

  '''
  runs bayesian clustering for number of clusters given by the list nclusters
  range. sr_bias_cd, centre_sd and outlier_sd_factors are used in determining
  model priors.
  '''

  if sample_args is None:
    sample_args = {}

  sample_args = dict(default_sample_args, **sample_args)

  with open(f"{cdir}/outputs/tables","rb") as fp:
    titre_table,lower,upper = pickle.load(fp)

  if kmeans_path is not None:
    with open(kmeans_path,"rb") as fp:
      kmeans_output = pickle.load(fp)

  rvec = np.array([1, 1, 1, -1, -1, -1])
  outlier_mean = np.array([0 for _ in range(len(antigens))])
  outlier_sd = 1.6*outlier_sd_factor

  if not os.path.exists(f"{cdir}/outputs/bayesian/log"):
    with open(f"{cdir}/outputs/bayesian/log", "w", encoding="utf-8") as fp:
      fp.write(str(datetime.now()) + '\n')

  for _, nclusters in enumerate(nclusters_range):

    if kmeans_output is not None:
      centres_kmeans = kmeans_output[nclusters]["cluster_centres"]
    else:
      centres_kmeans = None

    prior_params={
      "mu_sigma":centre_sd,
      "sr_bias_sd":sr_bias_sd
      }

    model, meta=\
      censored_titre_clustering_model(titre_table, nclusters, lower=lower.values,
                                      upper=upper.values, mu_ests = centres_kmeans,
                                      rvec=rvec, outlier_sd=outlier_sd,
                                      outlier_mean=outlier_mean,
                                      prior_params=prior_params
                                      )

    with model:
      idata = pm.sample(**sample_args, initvals=meta["initvals"])

    divergences = np.array(idata.sample_stats['diverging']).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
      r = az.rhat(idata)

    rhats = np.sum([np.count_nonzero(np.array(r[var])>1.01) for var in r.data_vars])

    with open(f"{cdir}/outputs/bayesian/log", "a", encoding="utf-8") as fp:
      fp.write(f"nclusters: {nclusters}, outlier_sd_factor:{outlier_sd_factor}"
               f", centre_sd: {centre_sd}, sr_bias_sd: {sr_bias_sd}\n")
      fp.write(f"divergences: {divergences}\n")
      fp.write(f"rhats: {rhats}\n")
      fp.write("\n")

    with open(f"{cdir}/outputs/bayesian/idata{nclusters}","wb") as fp:
      pickle.dump([idata,meta], fp)


def _main():

  if not os.path.exists(f"{cdir}/outputs/bayesian/"):
    os.mkdir(f"{cdir}/outputs/bayesian/")

  run_bayesian_clustering(range(1, 10),
                          kmeans_path=f"{cdir}/outputs/kmeans/combined")

if __name__ == "__main__":

  _main()
