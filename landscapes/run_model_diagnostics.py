#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:02:30 2024

@author: avicenna
"""

# pylint: disable=bad-indentation wrong-import-position import-error

import pickle
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from arviz import InferenceData
from xarray import DataArray, Dataset

sys.path.append(f"{cdir}/")
from run_landscapes import _get_model

sys.path.append(f"{cdir}/lib")
from landscapes import sample_observables

sys.path.append(f"{cdir}/../")
from common_utils import antigens


def _main(ncones_range=None, groups_range=None):

  if ncones_range is None:
    ncones_range = [2, 1, 1]
  if groups_range is None:
    groups_range = [0, 1, 2]

  assert len(groups_range) == len(ncones_range)


  nclusters = 3

  for ncones, i0 in zip(ncones_range, groups_range):

    with open(f"{cdir}/outputs/group{i0}_ncones{ncones}","rb") as fp:
      idata, meta_data, _ = pickle.load(fp)

    model, _, _ =\
      _get_model(nclusters=nclusters,  i0=i0, ncones=ncones,
                 **meta_data["model_args"])


    with model:
      pm.compute_log_likelihood(idata, extend_inferencedata=True)
      pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    sample_observables(model, idata, meta_data)

    if i0==2:
      coords = antigens[1:]
    else:
      coords = antigens[1:] + antigens[0:1]

    ax = az.plot_trace(idata["predictions"], var_names=["ag_mean_titres"],
                       coords={"ag":coords})

    ax[0,0].set_ylim(-0.5, 6.5)
    ax[0,0].set_xlim(-3, 10)
    ax[0,1].set_ylim(-2, 9)

    ax[0,0].set_xticks(range(-3, 11, 2))
    ax[0,1].set_yticks(range(-2,10,3))


    ax[0,0].set_title(f"Mean Antigen Titres Group {i0+1} Posteriors",
                      fontsize=15)
    ax[0,1].set_title(f"Mean Antigen Titres Group {i0+1} Traces",
                      fontsize=15)

    ax[0,0].set_xlabel("Value", fontsize=15)
    ax[0,1].set_xlabel("Draw", fontsize=15)

    ax[0,0].set_ylabel("Density", fontsize=15)
    ax[0,1].set_ylabel("Value", fontsize=15)
    ax[0,0].get_figure().tight_layout(w_pad=3)
    ax[0,0].get_figure().savefig(f"{cdir}/plots/diagnostics/trace_group{i0}_ncones{ncones}")


    lower = idata["constant_data"]["lower"].to_numpy()
    upper = idata["constant_data"]["upper"].to_numpy()
    obs_samples = idata.observed_data.obs.to_numpy()


    # note that loo_pit can not deal with thresholded values
    # correctly and they must be removed. see:
    # https://discourse.pymc.io/t/censored-linear-regression-relatively-good-ppc-horrible-loo-pit/14579/7
    # we do keep them for posterior predictive checks though

    fig, axes = plt.subplots(2, 1, figsize=(5,10))

    az.plot_ppc(idata, num_pp_samples=1000, ax=axes[0],
                kind="cumulative")
    axes[0].set_xlabel("Observed", fontsize=15)
    axes[0].set_ylabel("Cumulative Probability Density", fontsize=15)

    I = np.argwhere((lower != obs_samples) & (upper != obs_samples)).flatten()
    obs_names = idata.observed_data["obs_name"][I]
    pp_samples = idata.posterior_predictive.obs.copy()
    obs_samples = idata.observed_data.obs.copy()
    log_likelihood = idata.log_likelihood["obs"].copy()

    pp_subset = Dataset({"obs":DataArray(pp_samples.sel({"obs_name":obs_names}))})
    obs_subset = Dataset({"obs":DataArray(obs_samples.sel({"obs_name":obs_names}))})
    log_subset = Dataset({"obs":DataArray(log_likelihood.sel({"obs_name":obs_names}))})

    idata_subset = InferenceData()
    idata_subset.add_groups({"posterior_predictive":pp_subset})
    idata_subset.add_groups({"observed_data":obs_subset})
    idata_subset.add_groups({"log_likelihood":log_subset})
    idata_subset.add_groups({"posterior":idata.posterior.copy()})
    az.plot_loo_pit(idata_subset, y="obs", use_hdi=True, ax=axes[1])
    axes[1].set_ylim(0, 2.5)

    axes[1].set_xlabel("PIT", fontsize=15)
    axes[1].set_ylabel("Proability Density", fontsize=15)

    if i0>0:
      axes[0].legend().remove()
      axes[1].legend().remove()


    fig.tight_layout(w_pad=5)
    fig.savefig(f"{cdir}/plots/diagnostics/ppc_loo_ncones{ncones}_group{i0}.png")

    if i0==0:

      # group 0 is large enough to seperate into individual antigens

      fig,axes = plt.subplots(2, len(antigens), figsize=(5*len(antigens),10))
      for inda, antigen in enumerate(antigens):

        dims = [x for x in idata_subset["posterior_predictive"]["obs_name"].to_numpy() if
                antigen in x]

        subdata = idata_subset["posterior_predictive"].sel({"obs_name":dims})
        subobs = idata_subset["observed_data"].sel({"obs_name":dims})
        log_subset = Dataset({"obs":DataArray(log_likelihood.sel({"obs_name":dims}))})

        idata_subset2 = InferenceData()
        idata_subset2.add_groups({"posterior_predictive":subdata})
        idata_subset2.add_groups({"observed_data":subobs})
        idata_subset2.add_groups({"log_likelihood":log_subset})
        idata_subset2.add_groups({"posterior":idata.posterior.copy()})

        az.plot_ppc(idata_subset2, num_pp_samples=1000, ax=axes[0,inda],
                    kind="cumulative")
        az.plot_loo_pit(idata_subset2, y="obs", ax=axes[1,inda],
                        use_hdi=True)

        axes[1,inda].set_ylim(0, 2.5)

        axes[0, inda].set_xlabel("Observed", fontsize=15)
        axes[0, inda].set_ylabel("Probability Density", fontsize=15)

        axes[1, inda].set_xlabel("PIT", fontsize=15)
        axes[1, inda].set_ylabel("Probability Density", fontsize=15)

        if inda>0:
          axes[0, inda].legend().remove()
          axes[0, inda].set_ylabel('')
          axes[0, inda].set_xlabel('')
          axes[1, inda].set_ylabel('')
          axes[1, inda].set_xlabel('')

      fig.tight_layout(w_pad=5, h_pad=5)
      fig.savefig(f"{cdir}/plots/diagnostics/ppc_loo_ncones{ncones}_group{i0}_perag.png")


if __name__ == "__main__":

  _main()
