#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:22:45 2024

@author: avicenna
"""

# pylint: disable=bad-indentation import-error wrong-import-position

import pickle
import sys
import os

import arviz as az
import numpy as np

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/lib/")
sys.path.append(f"{cdir}")
from run_landscapes import _get_model
from landscapes import sample_grid_landscape
from visual import plot_landscape

sys.path.append(f"{cdir}/../")
from common_utils import _HDI, antigens


def _main(gw=0.2, ncones_range=None, groups_range=None):

  if ncones_range is None:
    ncones_range = [2, 1, 1]
  if groups_range is None:
    groups_range = [0, 1, 2]

  assert len(groups_range) == len(ncones_range)

  nclusters = 3

  r = np.array([-0.035, -1.85, 0.15])
  camera = {"up":{'x':0, 'y':0, 'z':1},
            "center":{'x':-0.035, 'y':0, 'z':0.15},
            "eye":{'x':r[0], 'y':r[1], 'z':r[2]}}

  aspect_ratio = [1, 1, 0.6]
  names = ["mean", "low", "high"]
  tI = [0,2,3]
  shifts = {"Delta":{"xshift":-20}, "Beta":{"yshift":-5, "xshift":5}}


  for ncones,i0 in zip(ncones_range, groups_range):

      with open(f"./outputs/group{i0}_ncones{ncones}","rb") as fp:
        idata, meta_data, _ = pickle.load(fp)

      model, _, _=\
       _get_model(nclusters=nclusters,  i0=i0, ncones=ncones,
                 **meta_data["model_args"])


      grid=\
        sample_grid_landscape(model, idata, meta_data["flat"], gw, "gs",
                              meta_data["base_map"]["ag_coordinates"])

      meta_data["grid"] = grid

      combined_landscape_means = az.summary(idata, group="predictions",
                                            var_names=["combined_grid_landscape_values"],
                                            hdi_prob=_HDI)

      x = np.array(sorted(list({p[0] for p in meta_data["grid"]})))
      y = np.array(sorted(list({p[1] for p in meta_data["grid"]})))

      figsize=(440, 440)

      impulses = az.summary(idata, var_names=["observed_gmts_wosrbias"],
                            hdi_prob=_HDI, group="predictions")

      I = [ind for ag in antigens for ind,label in enumerate(impulses.index)
           if ag.lower() in label.lower()]
      impulses = impulses.iloc[I,0].values

      if impulses.size != len(antigens):
        impulses = np.array([np.nan] + list(impulses))

      for j in range(3):


        fig=\
        plot_landscape(combined_landscape_means.iloc[:,tI[j]], x, y,
                       meta_data["base_map"], max_z=10, lld=0,
                       figsize=figsize, camera=camera, impulses=impulses,
                       antigen_label_shifts=shifts,
                       put_ticks=True, xbuffer=1, ybuffer=2,
                       aspect_ratio=aspect_ratio)
        fig.write_image(f"./plots/landscapes/ncones{ncones}_group{i0}_{names[j]}.png",
                        scale=2)
        fig.write_html(f"./htmls/ncones{ncones}_group{i0}_{names[j]}.html")


if __name__ == "__main__":

  _main(gw=0.2)
