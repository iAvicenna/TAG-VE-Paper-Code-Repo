#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:09:30 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import shutil
import sys
import os

from os import listdir
from os.path import isfile, join

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cdir)
from plot_utils import combine_images


source_dir = f"{cdir}/../misc/plots/"
target_dir = f"{cdir}/plots/SI"

shutil.copy(f"{source_dir}/sequences_plot.png",
            f"{target_dir}/figS24_sequences_plot.png")

onlyfiles = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]

assay_types = ["MN", "PRNT", "FRNT", "PNT"]
figs = []
for at in assay_types:

  files = [f"{source_dir}/{x}" for x in onlyfiles if '_'+at in x]

  figs += files

fig = combine_images(figs, 6, 2, xpad=100)
fig.save(f"{target_dir}/figS22_linegrid.png")
