#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:39:06 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, import-error, wrong-import-position

# pylint: disable=expression-not-assigned
# all are stuff like [ax.member_function() for ax in axes] to format
# axes without code clutter


import sys
import os
import matplotlib.pyplot as plt
import arviz as az
import numpy as np

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/../../")

from common_utils import ag_colours, _code_to_common_name

code_to_assay_name = {
  "FRNT":"FRNT",
  "MN":"MicroNeut",
  "PRNT":"PRNT",
  "PNT":"PseudoVirusNeut"
  }


def dataset_offset_plot(idata, assay_type, lab_assays, axes, hdi):

  '''
  assay_type: MN, PRNT, FRNT or PNT
  lab_assays: combinations of lab_code + assay_type
  '''

  az.plot_violin(idata, ax=axes, shade=0.9, hdi_prob=hdi, quartiles=True)

  _format_dataset_offset_plot(axes, lab_assays, assay_type)

  axes[1].text(0.1, 0.9, code_to_assay_name[assay_type],
               transform = axes[0].transAxes, fontsize=15,
               clip_on=False, zorder=1000)


def assay_offset_plot(idata, assay_types, hdi):

  '''
  does a violin plot of the offsets for the four assay types
  assay_types: which ones to take from MN, PRNT, FRNT or PNT
               also used in ordering the plot
  '''

  fig,axes = plt.subplots(1, len(assay_types), figsize=(1*len(assay_types), 4),
                          sharey=True)
  fig.subplots_adjust(wspace=0)

  az.plot_violin(idata["predictions"]["assay_offsets_mu_centred"].\
                 sel({"assay_type":list(assay_types)}),
                 var_names=["assay_offsets_mu_centred"],
                 hdi_prob=hdi, quartiles=True, shade=0.9,
                 ax=axes)

  _format_assay_offset_violin(axes, [code_to_assay_name[x] for x in assay_types])


  return fig,axes


def pairwise_lab_comparison(idata, lab_assays, antigens):

  '''
  compares the offset removed titres of each dataset to each other
  (off-diagonal plots) and to fitted gmt without offset (diagonal plots)

  lab_assays: which lab_code + assay_type combinations to plot
  antigens: used for ordering antigens
  '''
  antigens = list(antigens)

  n = len(lab_assays)

  fig,axes = plt.subplots(n, n, figsize=(2*n, 2*n), sharex=True,
                          sharey=True)


  for ind0,code0 in enumerate(lab_assays):

    difs = az.summary(idata, var_names=f"gmt_{code0}_dif",
                      group="predictions")

    lab_code0 = '_'.join(code0.split('_')[:-1])

    ags0 = [coord.split('_')[-1] for coord in
            np.array(idata["predictions"]["obs_name"]) if code0 in coord]

    xvals = [antigens.index(ag) for ag in ags0]
    I = np.argsort(xvals)
    xvals = sorted(xvals)

    yvals = difs.iloc[I,0]
    err = [difs.iloc[I,0] - difs.iloc[I,2],
           difs.iloc[I,3] - difs.iloc[I,0]]

    colours = [ag_colours[a] for a in np.array(ags0)[I]]

    axes[ind0, ind0].errorbar(xvals, yvals, fmt='none', ecolor='black', capsize=5,
                              yerr=err, alpha=0.15, zorder=0)

    axes[ind0,ind0].plot(xvals, yvals, color="black", zorder=1, alpha=0.3,
                         linewidth=3)
    axes[ind0, ind0].scatter(xvals, yvals, facecolor=colours, alpha=0.5,
                             s=50, edgecolor="darkgrey")
    axes[ind0, ind0].text(0.2, 0.8, _code_to_common_name[lab_code0].rjust(0),
                          transform = axes[ind0, ind0].transAxes, fontsize=10,
                          clip_on=False, zorder=1000, rotation=45,
                          horizontalalignment="center", verticalalignment="center")
    axes[ind0, ind0].plot(range(-1, len(antigens)+1), np.zeros((len(antigens)+2,)),
                          alpha=0.2, color="black")
    axes[ind0, ind0].set_xlim(-0.5, len(antigens)-0.5)
    axes[ind0, ind0].set_xticks(range(len(antigens)))
    axes[ind0, ind0].grid("on", alpha=0.2)
    axes[ind0, ind0].set_ylim(-7,7)
    axes[ind0, ind0].set_yticks(range(-6,8,2))
    axes[ind0, ind0].set_xticklabels(antigens, rotation=90)
    [x.set_linewidth(2.5) for x in axes[ind0,ind0].spines.values()]

    for ind1, code1 in enumerate(lab_assays):

      if code0 == code1:
        continue

      ags1 = [coord.split('_')[-1] for coord in
              np.array(idata["predictions"]["obs_name"]) if code1 in coord]


      common_ags = [ag for ag in ags1 if ag in ags0]

      difs = az.summary(idata, var_names=f"{code0}_{code1}_dif",
                        group="predictions")

      lab_code1 = '_'.join(code1.split('_')[:-1])

      xvals = [antigens.index(ag) for ag in common_ags]
      I = np.argsort(xvals)
      xvals = sorted(xvals)
      common_ags = [common_ags[i] for i in I]

      yvals = difs.iloc[I,0]
      err = [difs.iloc[I,0] - difs.iloc[I,2],
             difs.iloc[I,3] - difs.iloc[I,0]]

      colours = [ag_colours[a] for a in common_ags]


      axes[ind0, ind1].errorbar(xvals, yvals, fmt='none', ecolor='black', capsize=5,
                                yerr=err, alpha=0.15, zorder=0)

      axes[ind0,ind1].plot(xvals, yvals, color="black", zorder=1, alpha=0.3,
                           linewidth=3)
      axes[ind0, ind1].scatter(xvals, yvals, facecolor=colours, alpha=0.5,
                           s=50, edgecolor="darkgrey")
      axes[ind0, ind1].text(0.075, 0.5, _code_to_common_name[lab_code0].rjust(0),
                            transform = axes[ind0, ind1].transAxes, fontsize=10,
                            clip_on=False, zorder=1000, rotation=90,
                            horizontalalignment="center", verticalalignment="center")
      axes[ind0, ind1].text(0.5, 0.05, _code_to_common_name[lab_code1].rjust(0),
                            transform = axes[ind0, ind1].transAxes, fontsize=10,
                            clip_on=False, zorder=1000,
                            horizontalalignment="center", verticalalignment="center")
      axes[ind0, ind1].plot(range(-1, len(antigens)+1), np.zeros((len(antigens)+2,)),
                            alpha=0.2, color="black")

      axes[ind0, ind1].set_xticklabels(antigens, rotation=90)
      axes[ind0, ind1].grid("on", alpha=0.2)



  return fig, axes


def titre_line_plot(idata, assay_type, lab_assays, antigens, ylims, axes,
                    use_raw=False):

  '''
  This has two modes: use_raw = False or True.

  If True, for each panel, scatter plots lab raw data - offset and
  a line plot of the fitted gmt (via Bayesian model)

  If False, for each panel, scatter plots lab raw data and a point
  estimate of the gmt titre

  lab_assays: lab_code + assay_type combinations to take. they must have
              the same assay type as assay_type
  antigens: determines the order of antigens in plotting
  assay_type: which assay_type the chosen labs belong to
  '''

  if not use_raw:
    gmts = az.summary(idata, group="predictions", coords={"antigen":antigens},
                      var_names="gmts_centred").iloc[:,[0,2,3]].values
  else:
    gmts = idata["constant_data"]["gmts_pt_est"].\
      sel({"antigen":antigens}).to_numpy()[:,None]


  error_matrix = []
  errors = np.nan*np.zeros(gmts.shape[0])

  assert all(f"_{assay_type.upper()}" in x.upper() for x in
             lab_assays)

  for ind_ax,lab in enumerate(lab_assays):

    coord = [f"{lab}_{ag}" for ag in antigens if f"{lab}_{ag}"
             in idata["observed_data"]["obs"].coords["obs_name"]]


    if not use_raw:
      obs = az.summary(idata, group="predictions",
                       coords={"obs_name":coord},
                       var_names="lab_titres_wo_offset").iloc[:,0].values


    else: # use observed data if the plot is for non-parametric estimates
      obs =\
        idata["observed_data"]["obs"].sel({"obs_name":coord}).to_numpy()


    lab_code = '_'.join(lab.split('_')[:-1])

    xvals = [antigens.index(x.split('_')[-1]) for x in coord]
    ag = [antigens[i] for i in xvals]
    colours = [ag_colours[a] for a in ag]

    if not use_raw:
      axes[ind_ax].errorbar(range(len(antigens)), gmts[:,0], fmt='none',
                            ecolor='black', capsize=5,
                            yerr=[gmts[:,0] - gmts[:,1],
                                  gmts[:,2] - gmts[:,0]],
                            alpha=0.15, zorder=0)

    axes[ind_ax].plot(range(len(antigens)), gmts[:,0], color="black",
                      zorder=1, alpha=0.3, linewidth=4)
    axes[ind_ax].scatter(xvals, obs, facecolor=colours,
                         s=200, edgecolor="black")
    axes[ind_ax].text(0.5, 0.05, _code_to_common_name[lab_code].rjust(0),
                      transform = axes[ind_ax].transAxes, fontsize=25,
                      clip_on=False, zorder=1000,
                      horizontalalignment="center")

    if "EVC2" not in lab_code and "MUI2" not in lab_code and "RLID" not in lab_code:
      errors = np.nan*np.zeros(gmts.shape[0])

      if use_raw: # if not bayesian model, mean center with respect to gmt
                     # when calculating error
        obs -= np.mean(obs) - gmts[xvals,0].mean()

      errors[xvals] = (gmts[xvals,0]-obs)
      error_matrix.append(errors)


    #lower, upper, observed is used for determining when a value is thresholded
    lower = idata["constant_data"]["lower"].sel({"obs_name":coord})
    upper = idata["constant_data"]["upper"].sel({"obs_name":coord})
    obs_unshifted = idata["observed_data"]["obs"].sel({"obs_name":coord})

    for i,(low,up, o) in enumerate(zip(lower,upper, obs_unshifted)):

      if o==low:
        h = -1
      elif o==up:
        h = 1
      else:
        h = 0

      if h!=0:

        if use_raw:
          yval = obs_unshifted[i]
        else:
          yval = obs[i]

        axes[ind_ax].arrow(xvals[i], yval, 0, h, zorder=2,
                           head_width=0.12, facecolor="black", clip_on=False)


  axes[1].text(0.1, 1.05, code_to_assay_name[assay_type],
               transform = axes[0].transAxes, fontsize=25,
               clip_on=False, zorder=1000)


  _format_titre_line_plots(axes, antigens, assay_type, ylims)

  return np.array(error_matrix)


def folddrops_plot(idata, antigens):
  '''
  a plot of fold drops from the fitted gmts

  antigens: determines the order of antigens for the plot
  '''

  fig,ax = plt.subplots(1, 1, figsize=(5,5))


  # sampling fold drops will cause warnings because of subtracting identical
  # samples along the diagonal
  with np.errstate(invalid="ignore"):
    fold_drops =\
    az.summary(idata, group="predictions",
               var_names="pairwise_fold_drops",
               coords={"antigen":antigens,
                       "antigen2":antigens[0]}).iloc[:,[0,2,3]].values
  max_val = int(np.ceil(np.max(fold_drops[:,0])))+1

  colours = [ag_colours[antigen] for antigen in antigens]

  ax.plot(antigens, fold_drops[:,0], color="black", linewidth=3,
          alpha=0.8, zorder=0)

  ax.scatter(antigens, fold_drops[:,0], color=colours, linewidth=1,
             marker='o', s=150, edgecolor="black", zorder=2)

  ax.errorbar(range(len(antigens)), fold_drops[:,0],fmt='none',
              ecolor='black', capsize=5,
              yerr=[fold_drops[:,0] - fold_drops[:,1],
                    fold_drops[:,2] - fold_drops[:,0]],
              alpha=0.3, zorder=0)

  ax.set_ylim([np.floor(max_val-8), max_val])

  ax.grid(True, axis="y", alpha=0.3)

  ax.set_xticks(range(len(antigens)))
  ax.set_xticklabels(antigens, rotation=90, fontsize=15)
  ax.set_xlim(-0.5, len(antigens)-0.5)
  ax.set_ylim([-7,0.5])
  ax.set_yticks(range(-7,1))
  ax.set_yticklabels((2**np.arange(7,-1,-1)).astype(int),
                     fontsize=15)

  return fig, ax


def pairwise_difference_plot(idata, obs_types, coord, var_name, ylabel):
  '''
  used to do a violin plot of pairwise differences for either
  fold drop titres or assay offsets

  coord: "antigen" or "assay_type"
  var_name: "pairwise_assay_mu_differences" or "pairwise_fold_drops"
  obs_types: antigen_names or assay_types
  ylabel: log2 Fold Drop or Assay Offset
  '''

  fig,axes = plt.subplots(1, len(obs_types),
                          figsize=(5*len(obs_types), 5))


  for ind_ax, obs in enumerate(obs_types):

    # sampling pairwise diffs will cause warnings because of subtracting
    # identical samples along the diagonal

    with np.errstate(invalid="ignore"):
      diffs =\
        az.summary(idata, group="predictions",
                   var_names=[var_name],
                   coords={f"{coord}2":[obs],
                           f"{coord}":list(obs_types)}).iloc[:,[0,2,3]].values

    max_val = int(np.ceil(np.max(diffs[:,0])))+1

    if coord != "antigen":
      axes[ind_ax].plot(obs_types, diffs[:,0], color="black", linewidth=4,
                        alpha=0.5, marker='o', markersize=10)
    else:

      colours = [ag_colours[ag] for ag in obs_types]
      axes[ind_ax].plot(obs_types, diffs[:,0], color="black", linewidth=4,
                        zorder=-1, alpha=0.5)

      axes[ind_ax].scatter(obs_types, diffs[:,0], linewidth=1, zorder=2,
                           s=200, color=colours, edgecolor="black")


    axes[ind_ax].errorbar(range(len(obs_types)), diffs[:,0],fmt='none',
                          ecolor='black', capsize=5,
                          yerr=[diffs[:,0] - diffs[:,1],
                                diffs[:,2] - diffs[:,0]],
                          alpha=0.3, zorder=0)

    axes[ind_ax].set_ylim([np.floor(max_val-8), max_val])
    axes[ind_ax].plot(np.arange(-1, len(obs_types)+1),
                      [0 for _ in range(len(obs_types)+2)],
                      alpha=0.3, color="black")


  if coord=="assay_type":
    xticklabels = [code_to_assay_name[x] for x in obs_types]
  else:
    xticklabels = obs_types

  axes[0].set_ylabel(ylabel, fontsize=15)
  axes[0].set_xticks(range(len(obs_types)))
  axes[0].set_xticklabels(xticklabels, rotation=90, fontsize=15)

  [ax.set_xlim([-0.5, len(obs_types)-0.5]) for ax in axes]
  [ax.grid("on", alpha=0.3) for ax in axes]

  for ind_ax, ax in enumerate(axes[1:], start=1):
    xticklabels = ['']*ind_ax + [obs_types[ind_ax]] + ['']*(len(obs_types)-ind_ax-1)

    if coord=="assay_type":
      xticklabels = [code_to_assay_name[x] if x!='' else x for x in
                     xticklabels]


    axes[ind_ax].set_xticks(range(len(xticklabels)))
    axes[ind_ax].set_xticklabels(xticklabels, rotation=90, fontsize=15)


  return fig,axes


def _format_axis_boxes(axes, assay_type, label, remove_border=True):

  if assay_type != "MN":
    [plt.setp(ax.get_yticklabels(), visible=False) for ax in axes]
    [plt.setp(ax.get_xticklabels(), visible=False) for ax in axes]
    [_remove_ticks(ax) for ax in axes]
    [_remove_ticks(ax, "x") for ax in axes]
  else:
    axes[0].set_ylabel(label, fontsize=15)
    [plt.setp(ax.get_yticklabels(), visible=False) for ax in axes[1:]]
    [plt.setp(ax.get_xticklabels(), visible=False) for ax in axes[1:]]
    [_remove_ticks(ax) for ax in axes[1:]]
    [_remove_ticks(ax, "x") for ax in axes[1:]]

  if remove_border:
    [ax.spines['left'].set_visible(False) for ax in axes[1:]]
    [ax.spines['right'].set_visible(False) for ax in axes[:-1]]
  else:
    [ax.spines['left'].set_alpha(0.2) for ax in axes[1:]]
    [ax.spines['right'].set_alpha(0.2) for ax in axes[:-1]]


def _format_titre_line_plots(axes, antigens, assay_type, ylims):

  _format_axis_boxes(axes, assay_type, "log2 Titre", False)

  for _,ax in enumerate(axes):
    ax.set_xticks(range(len(antigens)))
    ax.set_xticklabels(antigens, fontsize=15,
                                 rotation=90)
    ax.set_xlim(-0.5, len(antigens)-0.5)
    ax.set_ylim(ylims)
    ax.set_yticks(range(ylims[0], ylims[1]+1, 1))
    yticklabels = 10*2.0**np.arange(ylims[0], ylims[1]+1,1)
    ax.set_yticklabels(yticklabels.astype(int))
    ax.grid(True, axis="y", alpha=0.2)


def _format_assay_offset_violin(axes, assay_names):

  for indax, ax in enumerate(axes):
    ax.set_title("")
    ax.set_xticks([0])
    ax.set_xticklabels([assay_names[indax]], rotation=90, fontsize=15)

    ax.set_ylim([-5, 4])
    ax.set_yticks(range(-5,5))
    ax.set_yticklabels(range(-5,5), fontsize=13)

    xvals = range(-4,5)
    yvals = [0 for _ in xvals]
    ax.plot(xvals, yvals, color="black", alpha=0.2)

    ax.set_xlim([-1.5, 1.5])
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.2)

  axes[0].set_ylabel("Assay Offset", fontsize=15)


def _format_dataset_offset_plot(axes, lab_assays, assay_type):

  _format_axis_boxes(axes, assay_type, "Dataset Offset")

  for indax,ax in enumerate(axes):
    ax.set_title("")


    lab_code = '_'.join(lab_assays[indax].split('_')[:-1])
    name = _code_to_common_name[lab_code]
    ax.set_xticks([0])
    ax.set_xticklabels([name], rotation=90, fontsize=15)

    ax.set_ylim([-5, 4])
    ax.set_yticks(range(-5,5))
    ax.set_yticklabels(range(-5,5), fontsize=13)

    xvals = range(-4,5)
    yvals = [0 for _ in xvals]
    ax.plot(xvals, yvals, color="black", alpha=0.2)

    ax.set_xlim([-2, 2])
    ax.grid(False)
    ax.grid(True, axis="y", alpha=0.2)


def _remove_ticks(ax, which="y"):

  if which=='y':
    ticks = ax.yaxis.get_major_ticks()
  elif which=='x':
    ticks = ax.xaxis.get_major_ticks()
  else:
    raise ValueError(f"which must be x or y but was {which}")


  for tick in ticks:
    tick.tick1line.set_visible(False)
    tick.tick2line.set_visible(False)
    tick.label1.set_visible(False)
    tick.label2.set_visible(False)


def add_ag_legend(axes, y, ag_names):

  '''
  adds a legend of antigen names at the bottom of a plot

  ag_names: which ag to add, colours of markers will come from the ag_colours
  in common_utils.py
  '''

  naxes = len(axes)
  mid_ax = int(np.floor(naxes/2))-1

  if axes.ndim==2:
    mid_ax = (1,mid_ax)
  loc = "upper left"
  if naxes/2 == int(naxes/2):
    pos = [-2.25, y]
  else:
    pos = [-1.75, y]

  for ag_name in ag_names:
    color = ag_colours[ag_name]
    axes[mid_ax].scatter(np.nan, np.nan, facecolor=color, label=ag_name,
                         s=550, edgecolor="black")

  axes[mid_ax].legend(loc=loc, bbox_to_anchor=pos, ncol=len(ag_names),
                      fontsize=40, handletextpad=-0.3, columnspacing=0.8)
