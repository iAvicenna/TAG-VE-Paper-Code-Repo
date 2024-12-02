#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:23:51 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, import-error, wrong-import-position

import pickle
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd



sys.path.append(f"{cdir}/lib/")
from clustering import censored_titre_clustering_model, sample_observables,\
  censored_titre_clustering_likelihood
from visual import cluster_centre_plot, bayesian_silhoutte_plot,\
  bayesian_within_cluster_centres_comparison_plot, cluster_bar_plot,\
    cluster_subgroups_bar_plot
from matplotlib.ticker import AutoLocator

sys.path.append(f"{cdir}/../")
from common_utils import _HDI, threshold


colour_to_name = {"tab:red":"BA.1+", "tab:green":"3+",
                 "tab:orange":"Beta/Delta 2+",
                 "tab:blue": "2+",
                 "tab:gray": "1+",
                 "tab:brown":"0+"
                 }
colour_order = ["tab:gray", "tab:orange", "tab:blue", "tab:green",
               "tab:purple", "tab:red"]

def _extract_posterior(idata, var_name, shape, group=None):

  new_shape = list(shape) + [3]

  with np.errstate(divide="ignore", invalid="ignore"):
    summary = az.summary(idata, group=group,
                         var_names=var_name,
                         hdi_prob=_HDI).iloc[:,[2,0,3]].values

  return np.reshape(summary, new_shape)


def _make_plots(ncluster_range, outlier_sd_factor, centre_sd, sr_bias_sd):

  with open(f"{cdir}/data/sr_colours","rb") as fp:
    sr_name_to_colour = pickle.load(fp)

  with open(f"{cdir}/outputs/tables","rb") as fp:
    titre_table,lower,upper = pickle.load(fp)

  with open(f"{cdir}/outputs/kmeans/combined","rb") as fp:
    kmeans_output = pickle.load(fp)

  lower = lower.values
  upper = upper.values

  nsr = titre_table.shape[0]

  mean_obs_likelihoods = []
  mean_s_scores = []

  antigens = ['Alpha', 'Beta', 'Delta', 'BA.1', 'BA.5',  'XBB.1.5']

  fig1,ax1 = plt.subplots(1, len(ncluster_range),
                          figsize=(5*len(ncluster_range),5),
                          sharex=True, sharey=True, squeeze=False)

  ax_counter = 0
  xvals = []


  for _,nclusters in enumerate(ncluster_range):

    centres_kmeans = kmeans_output[nclusters]["cluster_centres"]

    with open(f"{cdir}/outputs/bayesian/idata{nclusters}","rb") as fp:
      idata, meta = pickle.load(fp)

    rvec = np.array([1, 1, 1, -1, -1, -1])


    outlier_mean = np.array([np.nanmean(titre_table) for _
                             in range(len(antigens))])
    outlier_sd = 1.6*outlier_sd_factor

    prior_params={
      "mu_sigma":centre_sd,
      "sr_bias_sd":sr_bias_sd
      }

    model, _=\
      censored_titre_clustering_model(titre_table, nclusters, lower=lower,
                                      upper=upper, mu_ests = centres_kmeans,
                                      rvec=rvec, outlier_sd=outlier_sd,
                                      outlier_mean=outlier_mean,
                                      prior_params=prior_params)

    obs_likelihoods=\
    censored_titre_clustering_likelihood(titre_table, nclusters, idata,
                                         upper=upper, lower=lower,
                                         rvec=rvec)

    assert idata["posterior"]["mu1"]["cluster"].size == len(model.coords["cluster"])

    N = nclusters + 1*int(nclusters>1) # when nclusters>1 there is an outliers cluster
    cluster_likelihoods = _extract_posterior(idata, "cluster_likelihoods",
                                             (N, nsr))

    with model:
      pm.compute_log_likelihood(idata, extend_inferencedata=True)

    sample_observables(idata, rvec, model)


    b = _extract_posterior(idata, "b", (nsr,))
    mu = _extract_posterior(idata, "mu", (nclusters, len(antigens)),
                           group="predictions")


    if nclusters>1:
      folddrop_differences = _extract_posterior(idata, "folddrop_differences",
                                                group="predictions",
                                                shape=(nclusters, nclusters, len(antigens)))
      folddrops = _extract_posterior(idata, "folddrops", group="predictions",
                                     shape=(nclusters, len(antigens)))

    xvals.append(nclusters)

    fig,ax = plt.subplots(nclusters, 6, figsize=(30, nclusters*5),
                          sharex=True, sharey=True)

    ax = az.plot_posterior(idata, group="predictions",
                           var_names=["mu"], ax=ax)

    for a in ax.flatten():
      [spine.set_visible(True) for spine in a.spines.values()]
      a.yaxis.set_major_locator(AutoLocator())
    fig.tight_layout()
    fig.savefig(f"{cdir}/plots/bayesian/mu_posterior{nclusters}.png")

    fig,ax = plt.subplots(2,1, figsize=(4,10), squeeze=False)

    ax = az.plot_trace(idata, axes=ax.T, var_names=["rank"], compact=True)
    ax[0,0].set_title("Rank Posteriors")
    ax[0,1].set_title("idata Plots")

    fig.tight_layout()
    fig.savefig(f"{cdir}/plots/bayesian/trace{nclusters}.png")

    mean_obs_likelihoods.append(obs_likelihoods.mean())
    colours = [sr_name_to_colour[key] for key in model.coords["obs"]]

    #red: new variant, green: 3 or more, blue 2, orange 1, none:gray

    if nclusters>1:
      cluster_centres = np.concatenate([mu[...,1],
                                        meta["outlier_mean"][None,:]])
    else:
      cluster_centres = mu[...,1]

    fig0,axes,order=\
    cluster_centre_plot(cluster_centres, titre_table.values,
                        cluster_likelihoods[...,1],
                        lower=mu[...,0], upper=mu[...,2],
                        bias=b[:,1], normalize=True, thresholdeds=[lower,upper],
                        colours=colours, threshold=threshold)

    axes[0,0].set_xticks(range(len(antigens)))
    for irow in range(axes.shape[0]):
      axes[irow,0].set_ylabel("Centred Log 2 titre", fontsize=15)
      for icol in range(axes.shape[1]):
        axes[irow, icol].set_xticks(range(len(antigens)))
        axes[irow, icol].set_xticklabels(antigens, rotation=45, fontsize=14,
                                         ha="right")

    _add_legend(axes[0,-1], colours)

    fig0.tight_layout(w_pad=4.65, rect=[0, 0, 0.99, 1])
    fig0.savefig(f"{cdir}/plots/bayesian/cluster_centre_plot{nclusters}.png")


    _, _, s_scores=\
      bayesian_silhoutte_plot(cluster_likelihoods[...,1],
                              nclusters+int(nclusters>1), ax=ax1[0,ax_counter],
                              order=order, bottom=270)
    ax_counter += 1
    mean_s_scores.append(s_scores.mean())


    if nclusters>1:
      fig2,ax2 =\
        bayesian_within_cluster_centres_comparison_plot(folddrop_differences,
                                                        folddrops, order=order)

      ax2[0,1].set_ylim([-7,6])
      ax2[0,1].set_yticks(range(-7,8,2))
      ax2[0,0].set_ylim([-7,6])
      ax2[0,0].set_yticks(range(-7,8,2))
      ax2[0,0].set_yticklabels(range(-7,8,2))

      ax2[0,0].set_ylabel("Log2 Fold Drop", fontsize=15)
      ax2[0,1].set_ylabel("Log2 Fold Drop Difference", fontsize=15)

      ax2[0,0].set_xticks(range(len(antigens)))
      ax2[0,0].set_xticklabels(antigens, rotation=90, fontsize=15)

      for a in ax2.flatten()[1:]:
        plt.setp(a.get_xticklabels(), visible=False)
        plt.setp(a.get_yticklabels(), visible=False)

      fig2.tight_layout(h_pad=0, w_pad=2)
      fig2.savefig(f"{cdir}/plots/bayesian/cluster_centre_comparison{nclusters}.png")


    fig3,axes3=\
    cluster_bar_plot(cluster_likelihoods[...,1], sr_name_to_colour,
                     colours, colour_to_name=colour_to_name, order=order,
                     threshold=threshold, colour_order=colour_order)

    axes3[0,-1].set_title("Whole Dataset", fontsize=15)
    for j in range(nclusters):
      axes3[0,j].set_title(f"Cluster {j+1}", fontsize=15)


    if nclusters>1:
      axes3[0,-2].set_title("Outliers Cluster", fontsize=15)

    fig3.tight_layout(w_pad=7.25, rect=[0, 0, 1, 1], pad=1.15)

    fig3.savefig(f"{cdir}/plots/bayesian/cluster_bar{nclusters}.png")

    plt.close("all")

    # make a plot where clusters encounter subgroups are shown
    meta_table = pd.read_csv(f"{cdir}/outputs/subsetted_table.csv", header=0,
                             index_col=None)

    if nclusters>1:
      fig,ax=\
        cluster_subgroups_bar_plot(cluster_likelihoods[...,1], meta_table,
                                   np.array(model.coords["obs"]), order)

      fig.savefig(f"{cdir}/plots/bayesian/cluster_subgroups_bar{nclusters}.png")


  fig1.tight_layout()
  fig1.savefig(f"{cdir}/plots/bayesian/bayesian_silhoutte_plot.png")

  fig4,ax4 = plt.subplots(1, 2, figsize=(10,5))
  ax4[0].plot(xvals, -np.array(mean_obs_likelihoods), marker='o',
              color="black")

  ax4[1].plot(xvals, mean_s_scores, color="black", marker='o')
  ax4[1].set_ylim([0,1.1])

  for i in range(2):
    ax4[i].grid("on", alpha=0.4)
    ax4[i].set_xlabel("# clusters")
    ax4[i].set_xticks(ncluster_range)

  ax4[0].set_ylabel("-log Likelihood(Data|Param)")
  ax4[1].set_ylabel("Silhouette Score")

  fig4.tight_layout(w_pad=5)
  fig4.savefig(f"{cdir}/plots/bayesian/score_comparison.png")


  plt.close("all")


def _add_legend(ax, colours):

  colours = [c for c in colour_order if c in colours]

  for c in colours:
    ax.plot(np.nan, np.nan, color=c, label=colour_to_name[c],
            linewidth=2)

  ax.plot(np.nan, np.nan, label="Cluster Centre",
          linewidth=2, color="black")

  ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.04),
            fontsize=15)


if __name__ == "__main__":

  _make_plots(range(1,10), outlier_sd_factor=4, centre_sd=1, sr_bias_sd=0.5)
