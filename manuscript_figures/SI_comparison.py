#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:10:10 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import shutil
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(cdir)
from plot_utils import combine_images, add_text

# Comparison Diagnostics
source_dir1 = f"{cdir}/../control_comparison/plots/diagnostics"
source_dir2 = f"{cdir}/../control_comparison/plots/posteriors"

target_dir = f"{cdir}/plots/SI"

#Fig 1
shutil.copy(f"{source_dir1}/trace_T.png", f"{target_dir}/figS1_trace.png")

#Fig 2
img = add_text(["A", "B"], [[0.001, 0.001], [0.5, 0.001]],
               img_path=f"{source_dir1}/ppc_T.png",
               font_size=30)
img.save(f"{target_dir}/figS2_ppc.png")

#Fig 3
paths = [f"{source_dir1}/pareto.png", f"{source_dir1}/pareto_outrem.png"]
img = combine_images(paths, 2, 1)
img.save(f"{target_dir}/figS3_pareto.png")

#Fig 4
path = f"{source_dir1}/model_compare.png"
shutil.copy(path, f"{target_dir}/figS4_compare.png")

#Fig 5
path = f"{source_dir2}/lab_comparison.png"
shutil.copy(path, f"{target_dir}/figS5_lab_compare.png")


#Fig 6
path = f"{source_dir2}/pairwise_folddrop_plots.png"
shutil.copy(path, f"{target_dir}/figS6_pairwise_folddrop.png")

#Fig 7
path = f"{source_dir2}/pairwise_assay_offset_mu_differences.png"
shutil.copy(path, f"{target_dir}/figS7_pairwise_dif.png")
