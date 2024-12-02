#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:21:07 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import pickle
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/../")
sys.path.append(f"{cdir}/../clustering/lib")
from common_utils import threshold, _HDI, antigens
from clustering import censored_titre_clustering_model, sample_observables

def _extract_posterior(idata, var_name, shape, group=None):

  new_shape = list(shape) + [3]

  with np.errstate(divide="ignore", invalid="ignore"):
    summary = az.summary(idata, group=group,
                         var_names=var_name,
                         hdi_prob=_HDI).iloc[:,[2,0,3]].values

  return np.reshape(summary, new_shape)

def _main():


  ref = 1
  shift = 1

  control_path = f"{cdir}/../control_comparison/outputs/control_folddrops.csv"

  control_folddrops = pd.read_csv(control_path, header=0, index_col=0)
  labels = [f"pairwise_fold_drops[{antigen}, {antigens[ref]}]" for antigen
            in antigens]
  fitted_folddrops_control = control_folddrops.loc[labels,"mean"].values

  with open(f"{cdir}/../clustering/outputs/kmeans/combined","rb") as fp:
    kmeans_output = pickle.load(fp)

  centres_kmeans = kmeans_output[3]["cluster_centres"]

  I = np.argmax([centres_kmeans[i,0]-centres_kmeans[i,-1] for i in range(3)])

  kmeans_folddrops = centres_kmeans[I,:] - centres_kmeans[I,ref]

  outlier_sd_factor = 4
  centre_sd = 1
  sr_bias_sd = 0.5


  kmeans_path=f"{cdir}/../clustering/outputs/kmeans/combined"

  with open(kmeans_path,"rb") as fp:
    kmeans_output = pickle.load(fp)

  with open(f"{cdir}/../clustering/outputs/tables","rb") as fp:
    titre_table,lower,upper = pickle.load(fp)
  nsr = titre_table.shape[0]


  i = 3
  centres_kmeans = kmeans_output[i]["cluster_centres"]

  with open(f"{cdir}/../clustering/outputs/bayesian/idata{i}", "rb") as fp:
    idata,meta = pickle.load(fp)

  rvec = np.array([1, 1, 1, -1, -1, -1])
  outlier_mean = np.array([np.nanmean(titre_table) for _
                           in range(len(antigens))])
  outlier_sd = 1.6*outlier_sd_factor
  prior_params={
    "mu_sigma":centre_sd,
    "sr_bias_sd":sr_bias_sd
    }

  model, meta=\
    censored_titre_clustering_model(titre_table, i, lower=lower.values,
                                    upper=upper.values, mu_ests = centres_kmeans,
                                    rvec=rvec, outlier_sd=outlier_sd,
                                    outlier_mean=outlier_mean,
                                    prior_params=prior_params
                                    )

  sample_observables(idata, rvec, model)
  N = i + 1*int(i>1)
  cluster_likelihoods = _extract_posterior(idata, "cluster_likelihoods",
                                          (N, nsr))[...,1]

  mu = _extract_posterior(idata, "mu", (i, len(antigens)),
                          group="predictions")
  cluster_centres = np.concatenate([mu[...,1],
                                    meta["outlier_mean"][None,:]])[2,...]


  labels = np.argmax(cluster_likelihoods, axis=0)
  second_p = np.sort(cluster_likelihoods, axis=0)[-2,:]

  I = (labels==2) & (second_p<threshold)
  data = titre_table.iloc[I,:].values
  lower = lower.values[I,:]
  J = lower==data
  data = data - data[:,ref][:,None]
  replaced_data = data.copy()
  replaced_data[J] = data[J] - shift

  mean_data = np.nanmean(data,axis=0)
  replaced_mean_data = np.nanmean(replaced_data, axis=0)

  observed_fold_drops = mean_data[ref] - mean_data
  replaced_observed_fold_drops = replaced_mean_data[ref] - replaced_mean_data

  fitted_folddrops = cluster_centres[ref] - cluster_centres


  fig,ax = plt.subplots(1, 1, figsize=(5,5))

  ax.plot(kmeans_folddrops, color="tab:brown", marker='o', label="k-means")
  ax.plot(-observed_fold_drops, color="black", marker='o', label="Observed")
  ax.plot(-replaced_observed_fold_drops, color="tab:red", marker='o',
          label=f"Modified Observed ({-shift})")
  ax.plot(-fitted_folddrops, color="tab:green", marker='o', label="Modelled")
  ax.plot(fitted_folddrops_control, color="tab:purple", marker='o', label="Modelled Control")
  ax.set_xticks(range(len(antigens)))
  ax.set_xticklabels(antigens, rotation=90, fontsize=15)
  ax.set_xlabel("Variant", fontsize=15)
  ax.set_ylabel(f"log2 Fold Change wr {antigens[ref]}", fontsize=15)
  ax.set_yticks(np.arange(2,-7,-1))
  ax.set_yticklabels([f"1/{int(2**-x)}" if x<0 else int(2**x) for x in range(2,-7,-1)])
  ax.set_ylim(-7,2)
  ax.grid("on", alpha=0.2)

  ax.legend()

  fig.savefig(f"{cdir}/plots/folddrops_comparison.png",
              bbox_inches="tight")


if __name__ == "__main__":

  _main()
