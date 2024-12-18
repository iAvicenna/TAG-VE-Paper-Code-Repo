#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:45:49 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

import shutil
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cdir)
from plot_utils import combine_images, add_text, add_margin

source_dir1 = f"{cdir}/../clustering/plots/bayesian/"
source_dir2 = f"{cdir}/../clustering/plots/kmeans/"
source_dir3 = f"{cdir}/../misc/plots"

target_dir = f"{cdir}/plots/SI"

start = 7

#Fig 1
img = add_text(["A","B"],
               [[0.01, 0.001], [0.51, 0.001]],
               margin={"left":30} ,
               img_path = f"{source_dir1}/score_comparison.png",
               font_size=30)
img.save(f"{target_dir}/figS{start+1}_score.png")


#Fig 2
paths = [f"{source_dir1}/cluster_centre_comparison4.png",
         f"{source_dir1}/cluster_centre_comparison5.png"
         ]
img = combine_images(paths, 1, 2, xoffset=50, xpad=150)
img = add_text(["A","B"], [[0.001, 0.001], [0.46, 0.001]],
               img=img)
img.save(f"{target_dir}/figS{start+2}_centre_comparison.png")

#Fig 3
shutil.copy(f"{source_dir1}/cluster_centre_comparison3.png",
            f"{target_dir}/figS{start+3}_centre3.png"
            )

#Fig 4
img = add_margin(f"{source_dir1}/trace3.png",
                 top=17)

imgs = [f"{source_dir1}/mu_posterior3.png", img]
img = combine_images(imgs, 1, 2, xoffset=50, xpad=190)
img = add_text(["A","B"], [[0.001, 0.001], [0.88, 0.001]],
               img=img)
img.save(f"{target_dir}/figS{start+4}_muposterior.png")

#Fig 5
imgs = [f"{source_dir3}/emphasized_cluster_centres_kmeans.png",
        f"{source_dir3}/emphasized_cluster_centres_bayesian.png"]
img = add_text(["A","B"], [[0.001, 0.001], [0.001, 0.55]],
               img=img)
img = combine_images(imgs, 2, 1, xoffset=50, xpad=190)
img.save(f"{target_dir}/figS{start+5}_emphasized_cluster_centres.png")

#Fig 6
shutil.copy(f"{source_dir1}/cluster_subgroups_bar3.png",
            f"{target_dir}/figS{start+6}_cluster_subgroups.png"
            )

#Fig 7
shutil.copy(f"{source_dir1}/cluster_centre_plot3.png",
            f"{target_dir}/figS{start+7}_cluster3.png"
            )


#Fig 8
shutil.copy(f"{source_dir3}/folddrops_comparison.png",
            f"{target_dir}/figS{start+8}_folddrops_comparison.png"
            )
