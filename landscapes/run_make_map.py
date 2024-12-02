#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:05:59 2024

@author: avicenna
"""

# pylint: disable=bad-indentation

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

cdir = os.path.dirname(os.path.realpath(__file__))

def _remove_tick_lines(ax):

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)


def _get_map_limits(coordinates):


    xlim = [np.floor(np.nanmin(coordinates[:,0])-0.75),
            np.ceil(np.nanmax(coordinates[:,0])+0.75)]
    ylim = [np.floor(np.nanmin(coordinates[:,1])-0.75),
            np.ceil(np.nanmax(coordinates[:,1])+0.75)]

    lims = {}
    lims['x'] = xlim
    lims['y'] = ylim

    return lims



def _main():

  with open(f"{cdir}/../landscapes/data/ag_data","rb") as fp:
    ag_data = pickle.load(fp)

  ag_data["WT"] = [np.array([-2.10329177,  1.34851774]),"#393b79"]

  coordinates = np.array([val[0] for val in ag_data.values()])

  lims = _get_map_limits(coordinates)

  lims['y'][0] -= 0.2
  lims['x'][1] += 0.1


  rx = lims['x'][1] - lims['x'][0]
  ry = lims['y'][1] - lims['y'][0]

  fig,ax = plt.subplots(1, 1, figsize=(rx, ry))

  for ag in ag_data:

    coordinate = ag_data[ag][0]
    colour = ag_data[ag][1]

    ax.scatter(coordinate[0], coordinate[1], c=colour, edgecolor="black",
               s=3500, alpha=0.5, zorder=100, linewidths=2)

  ax.grid("on", alpha=0.3)

  ax.set_xticks(np.arange(lims['x'][0], lims['x'][1]+1))
  ax.set_yticks(np.arange(lims['y'][0], lims['y'][1]+1))

  _remove_tick_lines(ax)


  offsets = [[0.1, -0.9],
             [-0.45, 0.65],
             [-0.9, -1],
             [-0.4, -1],
             [-0.5, -1],
             [-0.65, -1],
             [-0.9, 0.6],
             ]


  for ag,offset in zip(ag_data, offsets):

    coord = ag_data[ag][0]

    if ag=="XBB.1.5":
      name = "XBB.2" #map antigen is actually XBB.2
    else:
      name = ag

    ax.text(coord[0]+offset[0], coord[1]+offset[1], name, size=25, zorder = 100)


  ax.set_xlim(*lims['x'])
  ax.set_ylim(*lims['y'])

  fig.savefig(f"{cdir}/plots/antigenic_map.png")
  plt.close("all")

_main()
