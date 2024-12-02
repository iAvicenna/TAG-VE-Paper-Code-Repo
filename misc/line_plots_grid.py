#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:56:04 2023

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"{cdir}/../")

from common_utils import antigens, ag_colours, log, _code_to_common_name,\
  _code_to_info


def _get_variants(table):

  return [ag for ag in antigens if not all(isinstance(x, float) and
                                           np.isnan(x) for x in
                                           table.loc[:,ag].values)]


def _parse_sera(subtable, min_encounters_threshold=np.inf):

  sera = subtable.loc[:,"serum_long"]

  table_antigens = _get_variants(subtable)

  parsed_data = {serum:{} for serum in sera}

  parsed_data['antigens'] = table_antigens
  parsed_data["min_encounter_levels"] =\
    sorted(list(set(subtable.loc[:, "min_encounters"].values)))

  for index,serum in zip(subtable.index,sera):
    min_encounters = subtable.loc[index,"min_encounters"]
    infection = subtable.loc[index, "infection"]
    vaccination = subtable.loc[index, "vaccination"]

    if str(infection).lower() in ["uninfected", "nan", "unknown"]:
      ninf = 0
    else:
      ninf = infection.count(',')+1

    if str(vaccination).lower() in ["unvaccinated", "nan", "unknown"]:
      nvac = 0
    else:
      nvac = vaccination.count(',')+1


    assert ninf+nvac==min_encounters

    min_encounters = min(min_encounters, min_encounters_threshold)

    parsed_infections = [z for x in str(infection).split(',') for z
                         in str(x).split('+') if x not in ["uninfected", "nan"]]
    parsed_vaccinations = [z for x in str(vaccination).split(',')
                           for z in str(x).split('+') if x not in ["unvaccinated","nan"]]


    parsed_data[serum]["infection_levels"] = set(parsed_infections)
    parsed_data[serum]["ninfections"] = len(parsed_infections)

    parsed_data[serum]["vaccination_levels"] = set(parsed_vaccinations)
    parsed_data[serum]["nvaccinations"] = len(parsed_vaccinations)

    parsed_data[serum]["titres"] = subtable.loc[index, table_antigens]
    parsed_data[serum]["min_encounters"] = min_encounters



  return parsed_data



def _ghost_axis(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_xlim(0,6)
  ax.set_ylim(-1,9)
  ax.set_yticks(np.arange(-1,10,1))
  ax.set_yticklabels((10*2.0**np.arange(-1, 10, 1)).astype(int).astype(str),
                              fontsize=16,color="white")
  ax.set_ylabel("titre in log2 scale", fontsize=16, color="white")
  ax.set_xticks(range(6))
  ax.set_xticklabels(antigens, rotation=90, fontsize=16,
                     color="white")
  ax.tick_params(axis='x', colors='white')
  ax.tick_params(axis='y', colors='white')
  ax.set_axisbelow(True)

def _color(infections, vaccinations):

  combined = set(infections).union(vaccinations)


  if "XBB.1" in combined or "XBB.1.5" in combined:
    return ag_colours["XBB.1"]
  if "BA.5" in combined:
    return ag_colours["BA.5"]
  if "BA.1" in combined:
    return ag_colours["BA.1"]
  if "Delta" in combined:
    return ag_colours["Delta"]
  if "Beta" in combined:
    return ag_colours["Beta"]
  if "Alpha" in combined:
    return ag_colours["Alpha"]
  if "WT" in combined:
    return ag_colours["WT"]
  if "UV" in combined:
    return "tab:purple"

  return "grey"


def _parsed_data_to_lineplot_data(parsed_data):

  max_y = -np.inf
  min_y = np.inf

  xvals = parsed_data["antigens"]


  data = {min(level,4):{"markers":[], "colors":[], "yvals":[], "lims":[], "xvals":[],
                 "gmt":np.zeros(len(xvals)), "serum_names":[]} for level in
          parsed_data["min_encounter_levels"]}

  sera = [x for x in parsed_data.keys() if x not in ["antigens","min_encounter_levels"]]

  if len(sera)==1 and sera[0]=="NIBSC_21/338":
    return data

  for serum in sera:

    if serum == "NIBSC_21/338":
      continue

    serum_data = parsed_data[serum]
    titres = serum_data["titres"].values
    min_encounters = serum_data["min_encounters"]

    if min_encounters>3:
      min_encounters=4

    log_titres = np.array([log(x) for x in titres])

    data[min_encounters]["yvals"].append(log_titres)

    max_y = max(max_y, np.nanmax(log_titres))
    min_y = min(min_y, np.nanmin(log_titres))


    serum_markers = np.array(['v' if '<' in str(titre) else '^'
                              if '>' in str(titre) else 'o' for titre in titres])
    data[min_encounters]["markers"].append(serum_markers)

    color = _color(serum_data["infection_levels"], serum_data["vaccination_levels"])
    data[min_encounters]["colors"].append(color)

    data[min_encounters]["gmt"] += log_titres
    data[min_encounters]["xvals"] = [antigens.index(x) for x in xvals]
    data[min_encounters]["serum_names"].append(serum)


  max_y = int(np.ceil(max_y))
  min_y = int(np.floor(min_y))

  for level in data:
    nsera = len(data[level]["serum_names"])
    data[level]["gmt"] /= nsera
    data[level]["lims"] = [min_y, max_y]

  return data


def _labsera_line_plots(plot_data, axes,  ylims=None, bias=0):

  if ylims is None:
    ylims = [np.min([plot_data[level]["lims"][0] for level in plot_data]),
            np.max([plot_data[level]["lims"][1] for level in plot_data]),
            ]


  for indax,level in enumerate(plot_data):
    markers = plot_data[level]["markers"]
    colors = plot_data[level]["colors"]
    log_titres = plot_data[level]["yvals"]
    xvals = np.array(plot_data[level]["xvals"])
    serum_names = plot_data[level]["serum_names"]
    axes[indax].text(4,8,f"n={len(serum_names)}", fontsize=16)

    if level<4:
      axes[indax].set_title(f"min num exposures = {level}", fontsize=16)
    else:
      axes[indax].set_title(f"min num exposures > {3}", fontsize=16)

    for inds,_ in enumerate(serum_names):

      axes[indax].plot(xvals, log_titres[inds]-bias, color=colors[inds], linewidth=2,
                       alpha=0.5, zorder=0)


      for mstyle in ['v','^']:
        I = [ind for ind,marker in enumerate(markers[inds]) if marker==mstyle]
        if len(I)==0: continue

        axes[indax].scatter(xvals[I], log_titres[inds][I]-bias, facecolor=colors[inds],
                            marker=mstyle, alpha=0.5, zorder=0)

    axes[indax].set_xticks(range(len(antigens)))
    xticklabels = [antigen if inda in xvals else "" for inda,antigen in
                   enumerate(antigens)]
    axes[indax].set_xticklabels(xticklabels, rotation=90,
                                fontsize=16)
    axes[indax].grid("on", alpha=0.2, color="black")
    axes[indax].set_ylim(ylims)
    axes[indax].set_yticks(np.arange(ylims[0], ylims[1]+1, 1))
    axes[indax].set_yticklabels((10*2.0**np.arange(ylims[0], ylims[1]+1, 1)).\
                                astype(int).astype(str), fontsize=16)
    axes[indax].set_ylabel("titre in log2 scale", fontsize=16)


def labsera_line_plots(max_nencounters=4):


  meta_path = f"{cdir}/../data/lab_data.csv"

  meta_table = pd.read_csv(meta_path, header=0, index_col=None)

  index = [label for label in meta_table.index if meta_table.loc[label,"serum_long"]
           not in ["NIBSC_21/338"]]

  meta_table = meta_table.loc[index,:]

  lab_codes = list(_code_to_common_name.keys())
  common_names = [_code_to_common_name[code] for code in lab_codes]

  lab_codes = [lab_codes[i] for i in np.argsort(common_names)]

  for lab_code in lab_codes:

    cn = _code_to_common_name[lab_code]
    print(f"{cn}", end=" ")
    subtable = meta_table[meta_table.lab_code.isin([lab_code])]


    assay_type_levels = set(subtable.loc[:,"assay_type"].values)
    if len(assay_type_levels)==0:
      continue


    for assay_type in  _code_to_info[lab_code]:

      bias = 0
      parsed_data =\
        _parse_sera(subtable[subtable.assay_type.isin([assay_type])], 4)

      lineplot_data = _parsed_data_to_lineplot_data(parsed_data)

      naxis = len({min(x,4) for x in parsed_data["min_encounter_levels"]})

      if naxis==0:
        continue


      naxis = len(parsed_data["min_encounter_levels"])


      fig,axes = plt.subplots(1, max_nencounters+1, figsize=(5*(max_nencounters+1), 4.75))
      plot_axes = []

      min_encounter_levels = [x if x<4 else 4 for x in parsed_data["min_encounter_levels"]]

      name = _code_to_common_name[lab_code] + f" ({assay_type.upper()})"

      axes[0].text(0.1,7, name, fontsize=25, zorder=10, backgroundcolor="white")

      for indax in range(max_nencounters+1):
        if indax not in min_encounter_levels:
          _ghost_axis(axes[indax])

        else:
          plot_axes.append(axes[indax])

      _labsera_line_plots(lineplot_data, plot_axes,
                          ylims=[-1, 9], bias=bias)


      fig.tight_layout(w_pad=4)
      fig.savefig(f"{cdir}/plots/{lab_code}_{assay_type}.png")



if __name__ == "__main__":

  labsera_line_plots()
