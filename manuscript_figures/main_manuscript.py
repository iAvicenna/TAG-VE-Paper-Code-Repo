#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:07:08 2023

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error


import shutil
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cdir)

from plot_utils import combine_images, add_text, crop, add_border, resize

source_dir1 = f"{cdir}/../control_comparison/plots"
target_dir = f"{cdir}/plots/"


#Figure1
paths = [f"{source_dir1}/posteriors/raw_titre_plots.png",
         f"{source_dir1}/posteriors/bayes_titre_plots.png"]

fig = combine_images(paths, 2, 1)
fig = add_text(["A","B"], [[0.001, 0], [0.001 ,0.48]], fig, font_size=50)

fig.save(f"{target_dir}/main/fig1_titre_plots.png")


#Figure2
path = f"{source_dir1}/posteriors/folddrops.png"
shutil.copy(path, f"{target_dir}/main/fig3_folddrops.png")



#Figure3
paths = [
      f"{source_dir1}/posteriors/datasets_offsets_violin.png",
      f"{source_dir1}/posteriors/assay_offsets_violin.png"
         ]
fig = combine_images(paths, 1, 2, xoffset=30, xpad=50, yoffset=10)
fig = add_text(["A","B"], [[0.001, 0], [0.8, 0.001]], fig, font_size=40)

fig.save(f"{target_dir}/main/fig2_offsets.png")



#Figure4

source_dir2 = f"{cdir}/../landscapes/plots/"
source_dir3 = f"{cdir}/../clustering/plots/bayesian/"

imgs = []
for ncones in [1,2]:
  for i0 in range(3):

    if i0>0 and ncones==2 or i0==0 and ncones==1:
      continue

    img = crop(img_path=f"{source_dir2}/landscapes/ncones{ncones}_group{i0}_mean.png",
               x=[0, 1], y=[0.25, 1])

    img = resize(xscale=0.4485, yscale=0.4485, img=img)

    img = add_border(img=img,
                     border_rel_width=0.0025,
                     fill=(30,30,30))

    imgs.append(img)

imgs = [imgs[2], imgs[0], imgs[1]]
fig = combine_images(imgs, 1, 3, xpad=111, xoffset=66)

cl_centre_img = crop(img_path=f"{source_dir3}/cluster_centre_plot3.png", x=[0, 1],
                     y=[0, 0.5])


fig = combine_images([cl_centre_img, fig,
                    f"{source_dir3}/cluster_bar3.png"],3,1,
                     xoffset=50, ypad=60)

fig = add_text(["A","B","C"], [[0.001,0.001], [0.001, 0.35],
                               [0.001,0.66]], fig)

fig.save(f"{cdir}/plots/main/fig4_landscapes.png")
