#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:53:04 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position

# pylint: disable=no-member


import sys
import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm

cdir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import arviz as az
from sklearn.metrics import silhouette_samples, silhouette_score

sys.path.append(cdir)

from _utils import combine_encounters, _sort, _sortlabels

def bayesian_silhoutte_plot(cluster_likelihoods, n_clusters, bottom=None,
                            ax=None, colours=None, pad=None, order=None):

  '''
  for bayesian clustering silhoutte score of a point is defined as
  probability of belonging to your cluster - max probability of belonging to
  any other cluster. this function does a silhoutte plot using this
  silhoutte scoring for each data.
  '''

  n_clusters = cluster_likelihoods.shape[0]

  cohesion = np.max(cluster_likelihoods, axis=0)
  if n_clusters>1:
    seperation = np.sort(cluster_likelihoods, axis=0)[-2,:]
  else:
    seperation = np.zeros(cluster_likelihoods.shape[1])
  silhouette_scores = cohesion - seperation
  labels = np.argmax(cluster_likelihoods, axis=0)

  if isinstance(labels, list):
    n_data = len(labels)
  else:
    n_data = labels.size

  if colours is None:
    colours = [cm.nipy_spectral(float(i) / n_clusters)
              for i in range(n_clusters)]

  if ax is None:
    fig_height = max(n_data*2/200,10)
    pad = max(10, int(n_data/40))
    fig,ax = plt.subplots(1, 1, figsize=(10, fig_height),
                          sharex=True)
  else:
    fig = ax.get_figure()

  if pad is None:
    pad = 10

  xlimits = [np.min(silhouette_scores)-0.1, 1.1]
  ax.set_xlim(*xlimits)
  ax.set_xlabel("Silhoutte Score")
  start = pad
  mean_val = silhouette_scores.mean()
  mean_scores = []

  for i in range(n_clusters):

    I = labels==order[i]
    n = np.count_nonzero(I)

    if n==0:
      ax.text(xlimits[1]+0.025, start + pad/2, f"{i+1}", color="red")
      start += pad
      mean_scores.append(np.nan)
      continue

    yvals = range(start, start+n)
    J = np.argsort(silhouette_scores[I])[::-1]
    xvals = silhouette_scores[I][J]

    mean_scores.append(np.mean(xvals))

    ax.fill_betweenx(yvals, 0, xvals , color=colours[i],
                     zorder=0, alpha=0.7)

    start += n + pad
    ytext = np.mean(yvals)
    ax.text(xlimits[1]+0.025, ytext, f"{i+1}", color="black")

  if bottom is None:
    bottom = start

  for i in range(n_clusters):
    ax.scatter(mean_scores[i], bottom, color=colours[i], marker='x',
               clip_on=False)

  ax.set_yticks([])
  ax.set_yticklabels([])
  ax.plot([mean_val, mean_val],
          [0, start + 20*pad],
          linestyle='dashed', color="black")

  ax.set_ylim([bottom, 0])
  ax.set_xlim(min(-0.1, ax.get_xlim()[0]), 1.1)

  ax.set_xlabel("Silhouette Score Distribution")
  ax.set_ylabel("Cluster")

  return fig, ax, silhouette_scores


def silhoutte_plot(cluster_labels, n_clusters, X, ax = None, colours=None,
                   normalize=False, metric="euclidean"):
  '''
  source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
  '''


  if normalize:
    shift = np.tile(np.nanmean(X, axis=1)[:,None], (1, X.shape[1]))
  else:
    shift = np.zeros(X.shape)

  X = X.copy() - shift

  if colours is None:
    colours = [cm.nipy_spectral(float(i) / n_clusters)
              for i in range(n_clusters)]
  n_data = X.shape[0]

  if ax is None:
    fig_height = max(n_data*2/200,10)
    fig,ax = plt.subplots(1, 1, figsize=(10, fig_height),
                          sharex=True)
  else:
    fig = ax.get_figure()

  score = silhouette_score(X, cluster_labels, metric=metric)

  sample_silhouette_values = silhouette_samples(X, _sortlabels(list(cluster_labels.copy())),
                                                metric=metric)
  n_clusters = len(set(cluster_labels))
  y_lower = 12

  for i in range(n_clusters)[::-1]:
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    if size_cluster_i == 0:
      continue

    y_upper = y_lower + size_cluster_i
    colour = colours[i]

    median = ith_cluster_silhouette_values[int(size_cluster_i/2)]
    median_y = np.arange(y_lower, y_upper)[int(size_cluster_i/2)]

    ax.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=colour,
        edgecolor=colour,
        alpha=0.7,
    )

    ax.scatter(median, median_y, marker='x', color="black")

    dif = np.abs(ith_cluster_silhouette_values - score)
    I = [ind for ind,val in enumerate(dif) if
         np.abs(val - np.min(dif))<1e-4]
    if len(I)==0:
      I = -1
      yval = y_lower
    else:
      I = I[-1]
      yval = list(range(y_lower,y_upper))[I]

    if size_cluster_i>0:
      per = np.round(100-100*(I+1)/(size_cluster_i),2)
      ax.scatter(0, yval, marker='x', color="black")
      ax.text(-0.1, yval, f"%{per}")

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(1.125, y_lower + 0.5 * size_cluster_i, str(i+1))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 12  # 10 for the 0 samples

    ax.set_xlabel("Silhouette Score Distribution")
    ax.set_ylabel("Cluster")

  # The vertical line for average silhouette score of all the values
  ax.axvline(x=score, color="black", linestyle="--")
  ax.set_xlim(min(-0.2, ax.get_xlim()[0]), 1.1)
  ax.set_yticks([])
  ax.set_ylim(0, y_lower)

  return fig, ax


def mse_plot(errs, ax):

  '''
  mean squared error plot for kmeans
  '''

  max_err = 0
  max_nclusters = len(errs)

  ax.plot(errs, color="black", marker='o', markeredgecolor="black")
  max_err = max(max_err, np.max(errs)) + 0.15
  ax.set_xticks(range(max_nclusters))
  ax.set_xticklabels(range(1, max_nclusters+1))

  ax.set_ylim([0, max_err])

  ax.set_title("Within Cluster MSE (after Imputation)")
  ax.set_xlabel("Number of Clusters")
  ax.set_ylabel("Mean Squared Error")


def cluster_centres_posteriors(idatas, max_nclusters, ndims):

  '''
  posterior for the cluster centres obtained from bayesian clustering model
  idatas should contain the relevant inference data for each number of clusters
  '''

  figs = []
  for i in range(max_nclusters):
    fig,_ = _cluster_centre_posteriors(idatas[i], i+1, ndims)
    figs.append(fig)

  return figs


def _cluster_centre_posteriors(idata, nclusters, ndims):

  fig,axes = plt.subplots(nclusters, ndims, figsize=(5*ndims, 5*(nclusters)),
                          sharex=True, sharey=True, squeeze=False)

  az.plot_posterior(idata, var_names=["mu"], ax=axes, skipna=True)
  [spine.set_visible(True) for ax in axes.flatten() for spine in
   ax.spines.values()]

  return fig,axes


def cluster_centre_plot(centres, X, cluster_likelihoods, normalize=False,
                        threshold=0.1, lower=None, upper=None,
                        ylims=None, axes=None, order=None,
                        colours=None, bias=None, thresholdeds=None,
                        contains_outlier_cluster=True, thresholded_alpha=0.05,
                        thresholded_zorder=0):

  '''
  for cluster centres obtained from a bayesian clustering run, it plots
  each centre as well as the data that is identified as belonging to that
  cluster.

  it shows thresholded titres and the segments leading to them with as
  transparent which can be tweraked via thresholded_alpha parameter.
  thresholdeds must be supplied for this. this is a list with two elements
  first one lower thresholded values and second upper. both has the same
  shape as data. whereever data is equal to one of these, there is thresholding.

  lower and upper denote the HDI intervals for the centres, they should
  have the same shape as centres.

  The plot has two rows, first row shows data which has high certainty of
  belong to its cluster where as the second row shows low certainty. If
  the probability of confusion (max probability of belonging to any other
  cluster) is larger than threshold, then that data is plotted in the second
  row.

  last plot is the outlier cluster, if contains_outlier_cluster=True.

  '''

  if normalize:
    if thresholdeds is not None:
      thresholdeds[0] = thresholdeds[0].copy() - np.nanmean(X,axis=1)[:,None]
      thresholdeds[1] = thresholdeds[1].copy() - np.nanmean(X,axis=1)[:,None]
    X = X.copy() - np.nanmean(X,axis=1)[:,None]

  max_nclusters = centres.shape[0]
  ndims = centres.shape[1]

  if max_nclusters>1 and not all(x==1.00 for x in np.max(cluster_likelihoods,axis=0)):
    nrows = 2
  else:
    nrows = 1

  if axes is None:
    fig,axes = plt.subplots(nrows, max_nclusters,
                            figsize=(5*max_nclusters+3, 4.25*nrows),
                            sharey=True, squeeze=False)
  else:
    fig = axes.flatten()[0].get_figure()
    if axes.ndim==1:
      axes = np.array([axes])

  if order is None:
    if contains_outlier_cluster:
      order = sorted(range(max_nclusters-1), key = lambda x: _sortcentre(centres[x,:]))
      order = list(order) + [max_nclusters-1]
      nfinal = max_nclusters-1
    else:
      order = sorted(range(max_nclusters), key = lambda x: _sortcentre(centres[x,:]))
      nfinal = max_nclusters
  else:
    if contains_outlier_cluster:
      nfinal = max_nclusters-1
    else:
      nfinal = max_nclusters

  labels = np.argmax(cluster_likelihoods, axis=0)

  if colours is None:
    colours = np.array(["tab:grey" for _ in range(X.shape[0])])
  else:
    colours = np.array(colours)

  for i0 in range(max_nclusters):

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


    if max_nclusters>1:
      ax_inds = (sorted_likelihoods[-2,:]>threshold).astype(int)
    else:
      ax_inds = [0 for _ in range(yvals.shape[0])]


    if i0 != nfinal:
      if lower is not None and upper is not None:
        err = [centres[order[i0],:] - lower[order[i0],:],
               upper[order[i0],:] - centres[order[i0],:]]
        for irow in range(axes.shape[0]):
          axes[irow, i0].errorbar(range(ndims), centres[order[i0],:], alpha=1,
                                  yerr=err, color=[0.3,0.3,0.3], zorder=200, linewidth=3)
      else:
        axes[0, i0].plot(centres[order[i0],:],
                         color=[0.3,0.3,0.3], zorder=200, linewidth=3)

    for i in range(yvals.shape[0]):

      if ythresholdeds is None:
        markers = ['o' for _ in range(ndims)]
        alphas = [0.3 for _ in range(ndims)]
        zorders = [0 for _ in range(ndims)]
      else:
        #this should be a condition on equality
        markers = ['o' if ythresholdeds[0][i,j] != yvals[i,j] and
                   ythresholdeds[1][i,j] != yvals[i,j] else
                   'v' if ythresholdeds[0][i,j] == yvals[i,j] else
                   '^' for j in range(ndims)]
        alphas = [0.4 if ythresholdeds[0][i,j] != yvals[i,j] and
                  ythresholdeds[1][i,j] != yvals[i,j] else
                  thresholded_alpha if ythresholdeds[0][i,j] ==  yvals[i,j]
                  else thresholded_alpha for j in range(1,ndims)]


        zorders = [0 if ythresholdeds[0][i,j] != yvals[i,j] and
                   ythresholdeds[1][i,j] != yvals[i,j] else
                   thresholded_zorder if ythresholdeds[0][i,j] ==  yvals[i,j]
                   else thresholded_zorder for j in range(1,ndims)]


      for j in range(1,ndims):

        y = yvals[i,j-1:j+1] - b[i]

        axes[ax_inds[i], i0].plot([j-1,j], y, c=ycolours[i], zorder=zorders[j-1],
                                  linewidth=2, alpha=alphas[j-1], marker=None)

      for marker in ['o','v','^']:
        I = [ind_dim for ind_dim,m in enumerate(markers) if m==marker]
        if marker in ['v','^']:
          alpha = thresholded_alpha
          zorder = thresholded_zorder
        else:
          alpha = 0.4
          zorder = 0
        axes[ax_inds[i], i0].scatter(I, yvals[i,I] - b[i], color=ycolours[i],
                                     zorder=zorder, s=70, alpha=alpha,
                                     marker=marker)

    for irow in range(axes.shape[0]):
      N = np.count_nonzero(np.array(ax_inds)==irow)
      axes[irow, i0].grid("on", alpha=0.2)
      axes[irow, i0].text(0.05, 0.9, f"N={N}", fontsize=14,
                        transform=axes[irow,i0].transAxes)

  if ylims is None:
    ylims = axes[0,0].get_ylim()

  axes[0,0].set_ylim(np.floor(ylims[0]), np.ceil(ylims[1]))
  axes[0,0].set_yticks(range(int(np.floor(ylims[0])), int(np.ceil(ylims[1]))+2,
                           2))

  plt.close("all")



  return fig, axes, order


def cluster_subgroups_bar_plot(cluster_likelihoods, meta_table, sera, order,
                               threshold=0.2):

  '''
  like cluster_bar_plot but more detailed grouped for each cluster.
  not based on colour but based on meta_table this time. order determines
  the order of the cluster plotting.
  '''

  labels = np.argmax(cluster_likelihoods, axis=0)
  ratios = []

  for i0 in range(cluster_likelihoods.shape[0]):
    I = np.argwhere(labels==order[i0]).flatten()
    sorted_likelihoods = np.sort(cluster_likelihoods[:,I], axis=0)
    J = I[np.argwhere((sorted_likelihoods[-2,:]<threshold))].flatten()
    subsera = sera[J]
    m = meta_table[meta_table.serum_long.isin(subsera)]
    res = Counter(m.T.apply(combine_encounters))

    ratios.append(len(res.keys()) + 1)

  fig,ax = plt.subplots(1, len(ratios), figsize=(5*len(ratios), 7),
                        sharey=True,
                        width_ratios=ratios)


  for i0 in range(len(ratios)):

    I = np.argwhere(labels==order[i0]).flatten()
    sorted_likelihoods = np.sort(cluster_likelihoods[:,I], axis=0)
    J = I[np.argwhere((sorted_likelihoods[-2,:]<threshold))].flatten()
    subsera = sera[J]
    m = meta_table[meta_table.serum_long.isin(subsera)]
    res = Counter(m.T.apply(combine_encounters))

    keys = sorted(list(res.keys()), key=_sort)

    ax[i0].bar(range(len(res.keys())), [res[x] for x in keys],
               facecolor="darkgray", edgecolor="black")
    ax[i0].set_xticks(range(len(res.keys())))
    ax[i0].set_xticklabels(keys, rotation=90, fontsize=14)
    ax[i0].set_xlim([-1,len(keys)])
    ax[i0].set_yticks(range(0,24,2))
    ax[i0].set_ylim(0,24)


    if i0 != len(ratios)-1:
      ax[i0].set_title(f"Cluster {i0+1}", fontsize=16)
    else:
      ax[i0].set_title("Outliers", fontsize=16)

  ax[0].set_ylabel("Count", fontsize=15)

  fig.tight_layout()

  return fig,ax


def cluster_bar_plot(cluster_likelihoods, sr_name_to_colour, colours,
                     colour_to_name, threshold=0.7, axes=None, add_all=True,
                     order=None, colour_order=None, fs=None):

  '''
  Given a clustering, this constructs a bar plot for the type of sera
  in each cluster. Type is determined by sr_name_to_colour which assigns
  each serum in the data to a colour and the group name of each colour is
  given by colour_to_name. colours determine in which order to plot these.
  '''

  if fs is None:
    fs = [5,5]

  max_nclusters = cluster_likelihoods.shape[0]

  if add_all:
    ncols = max_nclusters+1
  else:
    ncols = max_nclusters

  if axes is None:
    fig,axes = plt.subplots(1, ncols,
                            figsize=(fs[0]*(ncols), fs[1]),
                            sharey=True, squeeze=False)
  else:
    fig = axes.flatten()[0].get_figure()
    if axes.ndim==1:
      axes = np.array([axes])

  if order is None:
    order = range(max_nclusters)

  labels = np.argmax(cluster_likelihoods, axis=0)

  if colour_order is None:
    colour_order = list(set(sr_name_to_colour.values()))

  colour_counts = Counter(sr_name_to_colour.values())
  colour_levels = [x for x in colour_order if x in colour_counts]
  xlabels = [colour_to_name[x] for x in colour_levels]

  colour_counts = [colour_counts[x] for x in colour_levels if x in colour_counts]

  if add_all:
    axes[0,-1].bar(xlabels, np.array(colour_counts)/len(sr_name_to_colour),
                   color=colour_levels, edgecolor="black", alpha=0.8)
    axes[0,-1].set_xticks(range(len(xlabels)))
    axes[0,-1].set_xticklabels(xlabels, rotation=45, fontsize=14,
                               ha="right")


  colours = np.array(colours)

  ax_counter=0
  sorted_likelihoods = np.sort(cluster_likelihoods, axis=0)

  if max_nclusters>1:
    high_p = (sorted_likelihoods[-2,:] < threshold)
  else:
    high_p = [True for _ in range(cluster_likelihoods.shape[1])]

  for i0 in range(max_nclusters):

    I = (labels==order[i0]) & (high_p)
    subset_colours = colours[I]


    colour_counts = Counter(subset_colours)
    colour_counts = [colour_counts[x] if x in colour_counts else 0 for x in colour_levels]

    if np.count_nonzero(I)>0:
      yval = np.array(colour_counts)/np.count_nonzero(I)
    else:
      yval = 0

    axes[0, ax_counter].bar(xlabels, yval, color=colour_levels,
                            edgecolor="black", alpha=0.8)
    axes[0, ax_counter].set_xticks(range(len(xlabels)))
    axes[0, ax_counter].set_xticklabels(xlabels, rotation=45, fontsize=14,
                                        ha="right")
    ax_counter += 1


  axes[0,0].set_xlabel("Encounters", fontsize=15)
  axes[0,0].set_ylabel("Proportion", fontsize=15)



  return fig, axes



def bayesian_within_cluster_centres_comparison_plot(folddrop_differences,
                                                    cluster_folddrops,
                                                    order=None):

  '''
  Given the folddrop differences of cluster centres obtained from
  the bayesian clustering model, it shows a grid plot of these pairwise
  differences betwee each cluster in the off-diagonal axes. the diagonal
  axes show the folddrops for each cluster.
  '''

  lower_folddrop_difs = folddrop_differences[...,0]
  mean_folddrop_difs = folddrop_differences[...,1]
  upper_folddrop_difs = folddrop_differences[...,2]

  lower_folddrops = cluster_folddrops[...,0]
  mean_folddrops = cluster_folddrops[...,1]
  upper_folddrops = cluster_folddrops[...,2]

  nclusters = lower_folddrop_difs.shape[0]
  ndims = lower_folddrop_difs.shape[-1]


  fig,axes = plt.subplots(nclusters, nclusters,
                          figsize=(5*nclusters, 5*nclusters),
                          squeeze=False)

  if order is None:
    order = np.arange(nclusters)

  for i0 in range(nclusters):

    for i1 in range(nclusters):

      if i0==i1:

        axes[i0,i0].sharex(axes[0,0])
        axes[i0,i0].sharey(axes[0,0])

        yvals = mean_folddrops[order[i0],:]
        err = [mean_folddrops[order[i0],:] -\
               lower_folddrops[order[i0],:],
               upper_folddrops[order[i0],:] -
               mean_folddrops[order[i0],:]]

      else:

        if i0>i1:
          axes[i0,i1].axis("off")
          continue

        axes[i0,i1].sharex(axes[0,1])
        axes[i0,i1].sharey(axes[0,1])

        yvals = mean_folddrop_difs[order[i0],order[i1],:]
        err = [mean_folddrop_difs[order[i0],order[i1],:] -\
               lower_folddrop_difs[order[i0],order[i1],:],
               upper_folddrop_difs[order[i0],order[i1],:] -
               mean_folddrop_difs[order[i0],order[i1],:]]

      axes[i0, i1].plot([-1, ndims+1],[0, 0], alpha=0.2, color="black")

      axes[i0,i1].plot(yvals, color="black")

      axes[i0, i1].errorbar(range(ndims), yvals,  yerr=err, color="black",
                            zorder=1, linewidth=2)

      axes[i0, i1].grid("on", alpha=0.2)
      axes[i0,i1].set_xlim([-0.1, ndims-0.9])



  return fig,axes


def _sortcentre(yvals):
  c = np.polyfit(range(len(yvals)), yvals, 1)
  return c[0]
