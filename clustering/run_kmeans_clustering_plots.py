#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:12:50 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import pickle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{cdir}/lib/")

from visual import cluster_centre_plot, silhoutte_plot, cluster_bar_plot,\
  mse_plot

colour_to_name = {"tab:red":"BA.1+", "tab:green":"3+",
                 "tab:purple":"Beta/Delta 2+",
                 "tab:blue": "2+",
                 "tab:orange": "1+",
                 "tab:gray":"0+"}

colour_order = ["tab:gray", "tab:orange", "tab:blue", "tab:green",
               "tab:purple", "tab:red"]

def _make_plots():

  with open(f"{cdir}/outputs/tables","rb") as fp:
    titre_table,lower,upper = pickle.load(fp)

  lower = lower.values
  upper = upper.values

  with open(f"{cdir}/outputs/kmeans/combined","rb") as fp:
    output = pickle.load(fp)

  with open(f"{cdir}/data/sr_colours","rb") as fp:
    sr_name_to_colour = pickle.load(fp)

  sera = titre_table.index
  sr_colours = [sr_name_to_colour[sr_name] for sr_name in sera]

  antigens = titre_table.columns
  errs = []

  fig1,axes1 = plt.subplots(1, len(output),
                          figsize=(5*len(output), 5))

  for ind_cluster in output:

    centres = output[ind_cluster]["cluster_centres"]
    labels = output[ind_cluster]["labels"]
    data = output[ind_cluster]["imputed_data"]
    errs.append(output[ind_cluster]["err"])

    n = centres.shape[0]
    cluster_likelihoods = np.zeros((n, titre_table.shape[0]))
    # dummy cluster centerlikelihoods normally used in bayesian but needed
    # here to get labels from in plotting functions

    for i in range(n):
      cluster_likelihoods[i, labels==i]=1

    if ind_cluster==3:
      order = [1,0,2]
    else:
      order = None

    centres = centres - centres.mean(axis=1)[:,None]
    fig0,ax0,order = cluster_centre_plot(centres, titre_table.values,
                                         cluster_likelihoods,
                                         normalize=True, colours=sr_colours,
                                         thresholdeds=[lower,upper],
                                         ylims=[-7,9],
                                         contains_outlier_cluster=False,
                                         thresholded_alpha=0.6,
                                         thresholded_zorder=1,
                                         order=order)


    ax0[0,0].set_xticks(range(len(antigens)))
    for irow in range(ax0.shape[0]):
      ax0[irow,0].set_ylabel("Centred Log 2 Titer", fontsize=15)

      for icol in range(ax0.shape[-1]):
        ax0[irow, icol].set_xticklabels([])

    ax0[-1,0].set_xticks(range(len(antigens)))
    ax0[-1,0].set_xticklabels(antigens, rotation=90, fontsize=14)
    ax0[-1,0].set_xlabel("Antigen", fontsize=15)
    _add_legend(ax0[0,-1], sr_colours)

    fig0.tight_layout(w_pad=4.65, rect=[0, 0, 0.99, 1])
    fig0.savefig(f"{cdir}/plots/kmeans/cluster_centres_nclusters{n}.png")


    if ind_cluster>1:
      silhoutte_plot(labels, ind_cluster, data,
                     ax=axes1[ind_cluster-1],  normalize=True)

    fig2,axes2=\
    cluster_bar_plot(cluster_likelihoods, sr_name_to_colour, sr_colours,
                     colour_to_name, order=order, add_all=False,
                     fs=[5,4.2], colour_order=colour_order)

    for j in range(n):
      axes2[0,j].set_title(f"Cluster {j+1}", fontsize=15)

    fig2.tight_layout(w_pad=5, rect=[0, 0, 1, 1], pad=1.15)
    fig2.savefig(f"{cdir}/plots/kmeans/cluster_centres_bars{n}.png")

  mse_plot(errs, axes1[0])
  fig1.tight_layout()
  fig1.savefig(f"{cdir}/plots/kmeans/silhoutte.png")
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
  _make_plots()
