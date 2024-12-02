#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:57:26 2024

@author: avicenna
"""
# pylint: disable=bad-indentation, wrong-import-position, import-error

import pickle
import sys
import os
cdir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
sys.path.append(f"{cdir}/lib/")

from diagnostics import PyMCLOOWrapper
from models import gmt_bayesian_model, sample_centred_observables
from matplotlib.ticker import AutoLocator

antigens = ['Alpha', 'Beta', 'Delta', 'BA.1', 'BA.5', 'XBB.1.5']

def _format_pareto_plot(ax):

  xvals = list(range(-10, 200,10))
  ax.set_ylim([0,2])
  ax.set_xlim([-5, 110])
  ax.plot(xvals, [0.7 for _ in range(len(xvals))], color="black",
          alpha=0.2)


def _format_posterior_plot(axes):

  axes = axes.flatten()

  for i in range(6):
    axes[i].set_title(f"{antigens[i]}", fontsize=15)

    if i==0:
      axes[i].set_xlabel("Value", fontsize=15)
      axes[i].set_ylabel("Density", fontsize=15)
    else:
      axes[i].set_xlabel("")
      axes[i].set_ylabel("")

    [spine.set_visible(True) for spine in axes[i].spines.values()]
    axes[i].yaxis.set_major_locator(AutoLocator())


def _format_trace_plot(axes, use_t):

  if use_t:
    names = ["GMT", "Noise SD", "Assay Offsets", "Nu"]
  else:
    names = ["GMT", "Noise SD", "Assay Offsets"]

  for i,name in enumerate(names):
    axes[i,0].set_title(f"{name} Posterior Distributions",
                        fontsize=15)
    axes[i,1].set_title(f"{name} Trace Plots",
                        fontsize=15)
    axes[i,0].set_ylabel("Density", fontsize=15)
    axes[i,0].set_xlabel("Value", fontsize=15)

    axes[i,1].set_ylabel("Value", fontsize=15)
    axes[i,1].set_xlabel("Draw", fontsize=15)


def _pareto_add_labels(khats, threshold, ax):

  xdata = np.arange(khats.size)
  idxs = xdata[khats > threshold]
  coord_labels = np.array(khats["obs_name"])
  for idx in idxs:
    label = coord_labels[idx]
    label = label.split('_')[0] + '\n' + label.split('_')[-1]
    label = label.replace(".1.5","")

    if label == "CUH\nXBB":
      offset = -2
    else:
      offset = 0

    ax.text(idx + offset, khats[idx], label, horizontalalignment="center",
            verticalalignment="bottom", fontsize=8)


def main(run_loo, plot_loo):
  '''
  this script carries out diagnostics on the models and produces some
  relevant plots. In particular:

  1- It does posterior predictive plots
  2- It runs probability integral transform plots
  3- It runs leave-one-out crossvalidation to compare models. when there
     are data with bad pareto values, it runs exact loo on these
  4- Posterior and trace plots for some of the priors
  5- Posterior plots for centred gmts

  '''

  thresholds = [0.7, 0.7]

  fig_c, axes_c = plt.subplots(1, 2, figsize=(15,5))

  for ind_c,remove_outlier in enumerate([False, True]):

    idatas = {}
    reloos = {}
    loos = {}

    fig_p, axes_p = plt.subplots(1, 2, figsize=(10,5))

    if not run_loo and plot_loo:
      with open(f"{cdir}/outputs/reloos" + remove_outlier*"_outrem", "rb") as fp:
        reloos = pickle.load(fp)

      with open(f"{cdir}/outputs/loos" + remove_outlier*"_outrem", "rb") as fp:
        loos = pickle.load(fp)

    for ind_p, use_t in enumerate([True, False]):

      name_str = "N"
      model_name = "Normal"
      var_names=["_gmts", "noise_sd", "_assay_offsets_mu"]

      if use_t:
        name_str = "T"
        model_name = "StudentT"
        var_names.append("nu")


      if remove_outlier:
        name_str += "_outrem"

      with open(f"{cdir}/outputs/fit_data_{name_str}","rb") as fp:
        meta, idata, flat_table = pickle.load(fp)

      idatas[name_str] = idata

      model,_ = gmt_bayesian_model(flat_table, use_t=use_t)

      with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        pm.compute_log_likelihood(idata, extend_inferencedata=True)

      if run_loo:
        loos[model_name]  = az.loo(idata.copy())

        # do exact leave one out cross validation for observations
        # with bad pareto values
        relooer = PyMCLOOWrapper(model=model,
                                 idata_orig=idata.copy(),
                                 sample_kwargs=meta["sample_args"])

        reloos[model_name] = az.reloo(relooer, loos[model_name],
                                      verbose=False)

      if plot_loo:
        khats = loos[model_name].pareto_k
        az.plot_khat(loos[model_name], threshold=None,
                     ax=axes_p[ind_p])

        _pareto_add_labels(khats, thresholds[ind_p], axes_p[ind_p])


        axes_p[ind_p].set_title(model_name, fontsize=15)
        _format_pareto_plot(axes_p[ind_p])

      model,_ = gmt_bayesian_model(flat_table, use_t=use_t) #reload model after loo
      idata_pred = sample_centred_observables(model, idata.copy())

      axes = az.plot_trace(idata_pred, var_names=var_names)

      _format_trace_plot(axes, use_t=use_t)
      axes[0,0].get_figure().tight_layout()
      axes[0,0].get_figure().savefig(f"{cdir}/plots/diagnostics/trace_{name_str}.png")

      fig,axes = plt.subplots(2, 3, figsize=(15,10), sharex=True,
                              sharey=True)
      az.plot_posterior(idata_pred, group="predictions",
                        coords={"antigen":antigens},
                        var_names="gmts_centred", ax=axes)
      _format_posterior_plot(axes)
      fig.tight_layout()

      fig,axes = plt.subplots(1,2, figsize=(20,5))
      az.plot_ppc(idata_pred, var_names=["obs"], num_pp_samples=1000,
                  ax=axes[0], mean=True, kind="cumulative")
      axes[0].set_xlim([-1,12])
      axes[0].set_xlabel("Observed", fontsize=15)

      az.plot_loo_pit(idata_pred, y="obs", ax=axes[1], use_hdi=True)
      axes[1].set_xlabel("Observed Integral Transform", fontsize=15)
      axes[1].set_ylim([0,2])

      for i in range(2):
        axes[i].set_ylabel("Density", fontsize=15)

      fig.tight_layout(w_pad=8, rect=[0.03, 0, 1, 1])
      fig.savefig(f"{cdir}/plots/diagnostics/ppc_{name_str}")

    if run_loo:
      with open(f"{cdir}/outputs/loos" + remove_outlier*"_outrem", "wb") as fp:
        pickle.dump(loos, fp)

      with open(f"{cdir}/outputs/reloos" + remove_outlier*"_outrem", "wb") as fp:
        pickle.dump(reloos, fp)

    if plot_loo:
      df_comp_loo = az.compare(reloos)
      az.plot_compare(df_comp_loo, figsize=(10,5),
                      ax=axes_c[ind_c])


      if ind_c==0:
        axes_c[ind_c].legend().remove()

      if ind_c>0:
        axes_c[ind_c].set_ylabel("")

      axes_c[ind_c].set_title("")

    fig_p.tight_layout(w_pad=5)
    fig_p.savefig(f"{cdir}/plots/diagnostics/pareto" +
                  remove_outlier*"_outrem" + ".png")

  fig_c.tight_layout(w_pad=5)
  fig_c.savefig(f"{cdir}/plots/diagnostics/model_compare.png")


if __name__ == "__main__":

  main(False, True)
