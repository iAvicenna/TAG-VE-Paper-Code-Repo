#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:17:09 2024

@author: avicenna
"""
# pylint: disable=bad-indentation, import-error, wrong-import-position

import pickle
import sys
import os

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
cdir = os.path.dirname(os.path.realpath(__file__))

def _order_lab_assays(lab_assays):

  '''
  order lab_assays so that WHO antigens and in-house viruses are
  next to each other for easier comparison
  '''

  lab_assays = sorted(lab_assays,
                      key = lambda x: ["EVC" not in x, "AHRI" not in x,
                                       "MUI" not in x, "2" in x])
  # put MUI and EVC in-house and WHO samples next to each other
  # 2 is for in-house samples

  return lab_assays


sys.path.append(f"{cdir}/lib/")
from models import gmt_bayesian_model, sample_centred_observables
from visual import dataset_offset_plot, assay_offset_plot, titre_line_plot,\
  add_ag_legend, pairwise_difference_plot, folddrops_plot,\
    pairwise_lab_comparison

sys.path.append(f"{cdir}/../")
from common_utils import _HDI

with open(f"{cdir}/outputs/fit_data_T","rb") as fp:
  meta, idata, flat_table = pickle.load(fp)

model,meta = gmt_bayesian_model(flat_table, use_t=True)

sample_centred_observables(model, idata)

with np.errstate(invalid="ignore"):
  folddrops = az.summary(idata, var_names="pairwise_fold_drops",
                         group="predictions")
  folddrops.to_csv(f"{cdir}/outputs/control_folddrops.csv")

dataset_offsets = az.summary(idata, group="predictions",
                             var_names="dataset_offsets_centred")

I = np.argsort(dataset_offsets.iloc[:,0].values)
index = dataset_offsets.index[I]

assay_types = ("MN", "PRNT", "FRNT", "PNT")
antigens = ['Alpha', 'Beta', 'Delta', 'BA.1', 'BA.5', 'XBB.1.5']



fig,all_axes = plt.subplots(1, len(index), figsize=(1*len(index), 4),
                            sharey=True)
all_axes = all_axes.flatten()
fig.subplots_adjust(wspace=0)
counter = 0


for inda,assay_type in enumerate(assay_types):

  lab_assays = [label.split('[')[-1].split(']')[0] for label in index
                if assay_type.lower() in label.lower()]

  if assay_type == "FRNT":
    lab_assays = _order_lab_assays(lab_assays)

  axes = all_axes[counter:counter+len(lab_assays)]
  counter += len(lab_assays)

  sub_idata = idata["predictions"]["dataset_offsets_centred"].\
    sel({"lab_assay":lab_assays})

  dataset_offset_plot(sub_idata, assay_type, lab_assays, axes,
                      _HDI)


fig.savefig(f"{cdir}/plots/posteriors/datasets_offsets_violin.png",
            bbox_inches="tight")
plt.close(fig)


fig, ax = assay_offset_plot(idata, assay_types, _HDI)
fig.savefig(f"{cdir}/plots/posteriors/assay_offsets_violin.png",
            bbox_inches="tight")
plt.close(fig)


fig,all_axes = plt.subplots(2, 11, figsize=(3*11, 5*2),
                            sharey=True)
all_axes = all_axes.flatten()
fig.subplots_adjust(wspace=0, hspace=0.6)
counter = 0

obs_coords = model.coords["obs_name"]
error_matrices = []

lab_assay_order = []

for inda,assay_type in enumerate(assay_types):

  lab_assays = [label.split('[')[-1].split(']')[0] for label in index
                if assay_type.lower() in label.lower()]

  if assay_type == "FRNT":
    lab_assays = _order_lab_assays(lab_assays)

  axes = all_axes[counter:counter+len(lab_assays)]
  counter += len(lab_assays)

  error_matrix=\
  titre_line_plot(idata, assay_type, lab_assays, antigens, [-1,11],
                  axes)

  lab_assay_order += lab_assays
  error_matrices.append(error_matrix)

combined_error_matrix = np.concatenate(error_matrices)
per_antigen_RMSDs = np.sqrt(np.nanmean(combined_error_matrix**2, axis=0))
RMSD = np.sqrt(np.nanmean(combined_error_matrix**2))
print(per_antigen_RMSDs)
print(RMSD)
print("")

add_ag_legend(all_axes[:11], -1.75, antigens)

[ax.axis("off") for ax in all_axes[counter:]]
plt.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.18)
fig.savefig(f"{cdir}/plots/posteriors/bayes_titre_plots.png")

fig,axes=\
pairwise_lab_comparison(idata, lab_assay_order, list(antigens))

fig.tight_layout()
fig.savefig(f"{cdir}/plots/posteriors/lab_comparison.png")


fig,all_axes = plt.subplots(2, 11, figsize=(3*11, 5*2),
                            sharey=True)
all_axes = all_axes.flatten()
fig.subplots_adjust(wspace=0, hspace=0.6)
counter = 0

obs_coords = model.coords["obs_name"]
error_matrices = []

for inda,assay_type in enumerate(assay_types):

  lab_assays = [label.split('[')[-1].split(']')[0] for label in index
              if assay_type.lower() in label.lower()]

  if assay_type == "FRNT":
    lab_assays = _order_lab_assays(lab_assays)

  axes = all_axes[counter:counter+len(lab_assays)]
  counter += len(lab_assays)

  error_matrix=\
  titre_line_plot(idata, assay_type, lab_assays, antigens, [-1,11],
                  axes, use_raw=True)
  error_matrices.append(error_matrix)


combined_error_matrix = np.concatenate(error_matrices)
per_antigen_RMSDs = np.sqrt(np.nanmean(combined_error_matrix**2, axis=0))
RMSD = np.sqrt(np.nanmean(combined_error_matrix**2))
print(per_antigen_RMSDs)
print(RMSD)
print("")

[ax.axis("off") for ax in all_axes[counter:]]
plt.subplots_adjust(left=0.03, right=0.98, top=0.85, bottom=0.13)
fig.savefig(f"{cdir}/plots/posteriors/raw_titre_plots.png")


fig, axes = pairwise_difference_plot(idata, antigens, coord="antigen",
                                     var_name="pairwise_fold_drops",
                                     ylabel="log2 Fold Drop")


fig.tight_layout(w_pad=3)
fig.savefig(f"{cdir}/plots/posteriors/pairwise_folddrop_plots.png")

fig, axes = pairwise_difference_plot(idata, assay_types, coord="assay_type",
                                     var_name="pairwise_assay_mu_differences",
                                     ylabel="Assay Offset")
fig.tight_layout(w_pad=3)
fig.savefig(f"{cdir}/plots/posteriors/pairwise_assay_offset_mu_differences.png")


fig,ax = folddrops_plot(idata, antigens)
ax.set_ylabel("Fold Drop", fontsize=15)
fig.tight_layout()
fig.savefig(f"{cdir}/plots/posteriors/folddrops.png")
