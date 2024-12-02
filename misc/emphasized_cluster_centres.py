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
import arviz as az


colour_to_name = {"tab:red":"BA.1+", "tab:green":"3+",
                 "tab:purple":"Beta/Delta 2+",
                 "tab:blue": "2+",
                 "tab:orange": "1+",
                 "tab:gray":"0+"}


cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/../clustering/lib/")

from clustering import censored_titre_clustering_model, sample_observables
colour_order = ["tab:gray", "tab:orange", "tab:blue", "tab:green",
               "tab:purple", "tab:red"]

sys.path.append(f"{cdir}/../")

from common_utils import _HDI, threshold


def _sortcentre(yvals):
  c = np.polyfit(range(len(yvals)), yvals, 1)
  return c[0]

def cluster_centre_plot(centres, X, cluster_likelihoods, normalize=False,
                        lower=None, upper=None, threshold=0.1,
                        ylims=None, axes=None, order=None,
                        colours=None, bias=None, thresholdeds=None,
                        contains_outlier_cluster=True):
  '''
  a specialized cluster_centre_plot function which is used to plot
  cluster centres where thresholded titres for some of the sera are
  emphasized.
  '''


  if normalize:
    if thresholdeds is not None:
      thresholdeds[0] = thresholdeds[0].copy() - np.nanmean(X,axis=1)[:,None]
      thresholdeds[1] = thresholdeds[1].copy() - np.nanmean(X,axis=1)[:,None]
    X = X.copy() - np.nanmean(X,axis=1)[:,None]

  max_nclusters = centres.shape[0]
  ndims = centres.shape[1]

  nrows = 1
  if contains_outlier_cluster:
    nfinal = max_nclusters-1
  else:
    nfinal = max_nclusters


  if axes is None:
    fig,axes = plt.subplots(nrows, nfinal,
                            figsize=(5*nfinal+3, 4.25*nrows),
                            sharey=True, squeeze=False)
  else:
    fig = axes.flatten()[0].get_figure()
    if axes.ndim==1:
      axes = np.array([axes])


  labels = np.argmax(cluster_likelihoods, axis=0)

  if colours is None:
    colours = np.array(["tab:grey" for _ in range(X.shape[0])])
  else:
    colours = np.array(colours)

  for i0 in range(nfinal):

    I = labels == order[i0]

    yvals = X[I, :]
    ycolours = colours[I]

    if thresholdeds is not None:
      ythresholdeds = [thresholdeds[0][I,:], thresholdeds[1][I,:]]
    else:
      ythresholdeds = None


    if bias is not None:
      b = bias[I]
    else:
      b = np.zeros((np.count_nonzero(I),))

    sorted_likelihoods = np.sort(cluster_likelihoods[:,I], axis=0)

    do_plot = (sorted_likelihoods[-2,:]<=threshold).astype(int)

    if i0 != nfinal:
      if lower is not None and upper is not None:
        err = [centres[order[i0],:] - lower[order[i0],:],
               upper[order[i0],:] - centres[order[i0],:]]
        for irow in range(axes.shape[0]):
          axes[irow, i0].errorbar(range(ndims), centres[order[i0],:], alpha=0.8,
                                  yerr=err, color=[0.3,0.3,0.3], zorder=10,
                                  linewidth=3)
      else:
        axes[0, i0].plot(centres[order[i0],:],
                         color=[0.3,0.3,0.3], zorder=10, linewidth=3,
                         alpha=0.8)

    for i in range(yvals.shape[0]):

      if do_plot[i] != 1:
        continue

      markers = ['o' if ythresholdeds[0][i,j] != yvals[i,j] and
                 ythresholdeds[1][i,j] != yvals[i,j] else
                 'v' if ythresholdeds[0][i,j] == yvals[i,j] else
                 '^' for j in range(ndims)]

      I =  np.argwhere(ythresholdeds[0][i,:] == yvals[i,:]).flatten()

      if ycolours[i] in ["tab:green"] and (len(I)>2 or (5 in I and 4 in I)):
        alpha = 1
        zorder = 2
      else:
        alpha = 0.5
        zorder = 1

      y = yvals[i,:] - b[i]

      axes[0, i0].plot(range(ndims), y, c=ycolours[i], linewidth=2,
                       alpha=alpha, marker=None, zorder=zorder)

      for marker in ['o','v','^']:
        I = [ind_dim for ind_dim,m in enumerate(markers) if m==marker]

        if zorder==2 and marker=='v':
          mzorder = 4
          edgecolour="black"
        elif ycolours[i] in ["tab:green"]:
          mzorder = 2
          edgecolour="none"
        else:
          mzorder = 1
          edgecolour="none"

        axes[0, i0].scatter(I, yvals[i,I] - b[i], color=ycolours[i],
                            zorder=mzorder, s=70, alpha=alpha,
                            marker=marker, edgecolor=edgecolour)

    for irow in range(axes.shape[0]):
      axes[irow, i0].grid("on", alpha=0.2)


  if ylims is None:
    ylims = axes[0,0].get_ylim()

  axes[0,0].set_ylim(np.floor(ylims[0]), np.ceil(ylims[1]))
  axes[0,0].set_yticks(range(int(np.floor(ylims[0])), int(np.ceil(ylims[1]))+2,
                           2))

  plt.close("all")


  return fig, axes, order


def _extract_posterior(idata, var_name, shape, group=None):

  new_shape = list(shape) + [3]

  with np.errstate(divide="ignore", invalid="ignore"):
    summary = az.summary(idata, group=group,
                         var_names=var_name,
                         hdi_prob=_HDI).iloc[:,[2,0,3]].values

  return np.reshape(summary, new_shape)



def _make_kmeans_plot():

  with open(f"{cdir}/../clustering/outputs/tables","rb") as fp:
      titre_table,lower,upper = pickle.load(fp)
  lower = lower.values
  upper = upper.values

  with open(f"{cdir}/../clustering/outputs/kmeans/combined","rb") as fp:
    output = pickle.load(fp)

  with open(f"{cdir}/../clustering/data/sr_colours","rb") as fp:
    sr_name_to_colour = pickle.load(fp)

  sera = titre_table.index
  sr_colours = [sr_name_to_colour[sr_name] for sr_name in sera]

  antigens = titre_table.columns


  for ind_cluster in [3]:

    centres = output[ind_cluster]["cluster_centres"]
    labels = output[ind_cluster]["labels"]

    order = sorted(range(3),
                   key = lambda x: _sortcentre(centres[x,[1,3,4]]))
    n = centres.shape[0]
    cluster_likelihoods = np.zeros((n, titre_table.shape[0]))
    # dummy cluster centerlikelihoods normally used in bayesian but needed
    # here to get labels from in plotting functions

    for i in range(n):
      cluster_likelihoods[i, labels==i]=1


    centres = centres - centres.mean(axis=1)[:,None]
    fig0,ax0,order = cluster_centre_plot(centres, titre_table.values,
                                         cluster_likelihoods,
                                         normalize=True, colours=sr_colours,
                                         thresholdeds=[lower,upper],
                                         ylims=[-7,9],
                                         contains_outlier_cluster=False,
                                         order = order)


    ax0[0,0].set_xticks(range(len(antigens)))
    for irow in range(ax0.shape[0]):
      ax0[irow,0].set_ylabel("Centred Log 2 titre", fontsize=15)
      for icol in range(ax0.shape[1]):
        ax0[irow, icol].set_xticks(range(len(antigens)))
        ax0[irow, icol].set_xticklabels(antigens, rotation=45, fontsize=14)


    _add_legend(ax0[0,-1], sr_colours)

    fig0.tight_layout(w_pad=4.65, rect=[0, 0, 0.99, 0.95])
    fig0.savefig(f"{cdir}/plots/emphasized_cluster_centres_kmeans.png")



def _make_bayesian_plots():

  ncluster_range = [3]
  outlier_sd_factor = 4
  centre_sd = 1
  sr_bias_sd = 0.5

  data_dir = "../clustering/"

  with open(f"{data_dir}/data/sr_colours","rb") as fp:
    sr_name_to_colour = pickle.load(fp)

  with open(f"{data_dir}/outputs/tables","rb") as fp:
    titre_table,lower,upper = pickle.load(fp)

  with open(f"{data_dir}/outputs/kmeans/combined","rb") as fp:
    kmeans_output = pickle.load(fp)

  lower = lower.values
  upper = upper.values

  nsr = titre_table.shape[0]

  antigens = ['Alpha', 'Beta', 'Delta', 'BA.1', 'BA.5',  'XBB.1.5']


  xvals = []

  for _,nclusters in enumerate(ncluster_range):

    centres_kmeans = kmeans_output[nclusters]["cluster_centres"]

    with open(f"{data_dir}/outputs/bayesian/idata{nclusters}","rb") as fp:
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


    assert idata["posterior"]["mu1"]["cluster"].size == len(model.coords["cluster"])

    N = nclusters + 1*int(nclusters>1) # when nclusters>1 there is an outliers cluster
    cluster_likelihoods = _extract_posterior(idata, "cluster_likelihoods",
                                            (N, nsr))



    sample_observables(idata, rvec, model)


    b = _extract_posterior(idata, "b", (nsr,))
    mu = _extract_posterior(idata, "mu", (nclusters, len(antigens)),
                            group="predictions")



    xvals.append(nclusters)


    colours = [sr_name_to_colour[key] for key in model.coords["obs"]]

    #red: new variant, green: 3 or more, blue 2, orange 1, none:gray

    if nclusters>1:
      cluster_centres = np.concatenate([mu[...,1],
                                        meta["outlier_mean"][None,:]])
    else:
      cluster_centres = mu[...,1]

    order = sorted(range(3),
                   key = lambda x: _sortcentre(cluster_centres[x,[1,3,4]]))

    fig0, ax0, _=\
    cluster_centre_plot(cluster_centres, titre_table.values,
                        cluster_likelihoods[...,1],
                        lower=mu[...,0], upper=mu[...,2],
                        bias=b[:,1], normalize=True, thresholdeds=[lower,upper],
                        colours=colours, threshold=threshold,
                        contains_outlier_cluster=True,
                        ylims=[-7,9], order = order)


    ax0[0,0].set_xticks(range(len(antigens)))
    for irow in range(ax0.shape[0]):
      ax0[irow,0].set_ylabel("Centred Log 2 titre", fontsize=15)
      for icol in range(ax0.shape[1]):
        ax0[irow, icol].set_xticks(range(len(antigens)))
        ax0[irow, icol].set_xticklabels(antigens, rotation=45, fontsize=14)

    ax0[-1,0].set_xlabel("Antigen", fontsize=15)

    _add_legend(ax0[0,-1], colours)

    fig0.tight_layout(w_pad=4.65, rect=[0, 0, 0.99, 1])

    fig0.savefig(f"{cdir}/plots/emphasized_cluster_centres_bayesian.png")

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

  _make_kmeans_plot()
  _make_bayesian_plots()
