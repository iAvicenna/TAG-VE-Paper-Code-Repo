#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:47:08 2024

@author: avicenna
"""
# pylint: disable=bad-indentation, wrong-import-position, import-error

import shutil
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(cdir)
from plot_utils import combine_images, add_border, crop, resize


start = 15

#Fig 1
source_dir1 = f"{cdir}/../landscapes/plots/landscapes"
source_dir2 = f"{cdir}/../landscapes/plots/diagnostics"
source_dir3 = f"{cdir}/../landscapes/plots/"

target_dir = f"{cdir}/plots/SI"


#Fig 1
paths = [f"{source_dir2}/ppc_loo_ncones{ncones}_group{i}.png"
         for ncones,i in zip([2,1,1],[0,1,2])]

fig = combine_images(paths, 1, 3)
fig.save(f"{target_dir}/figS{start+1}_ppclandscapes.png")



#Fig 2
shutil.copy(f"{source_dir2}/ppc_loo_ncones2_group0_perag.png",
            f"{target_dir}/figS{start+2}_ppclandscapeperag.png")


#Fig 3
paths = [f"{source_dir2}/trace_group{i}_ncones{ncones}.png"
         for ncones,i in zip([2,1,1],[0,1,2])]

fig = combine_images(paths, 3, 1)
fig.save(f"{target_dir}/figS{start+3}_tracelandscapes.png")

#Fig 5
shutil.copy(f"{source_dir3}/model_comparison.png",
            f"{target_dir}/figS{start+4}_model_comparison_landscapes.png")


imgs = []
ncones = [2,1,1]
order = [0,1,2]
for hdi in ["high","mean","low"]:
  for i0 in [0,1,2]:

    img_path = f"{source_dir1}/ncones{ncones[i0]}_group{order[i0]}_{hdi}.png"

    img = crop(img_path=img_path, x=[0, 1], y=[0.25, 1])

    img = resize(xscale=0.423*0.77, yscale=0.45*0.763, img=img)

    img = add_border(img=img,
                     border_rel_width=0.0025,
                     fill=(30,30,30))
    imgs.append(img)

fig = combine_images(imgs, 3, 3, xscale=1, yscale=1)
fig.save(f"{target_dir}/figS{start+5}_landscapes.png")

#Fig 6
shutil.copy(f"{source_dir3}/antigenic_map.png",
            f"{target_dir}/figS{start+6}_antigenic_map.png")
