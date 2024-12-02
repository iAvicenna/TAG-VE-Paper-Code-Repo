#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:55:17 2024

@author: avicenna
"""

# pylint: disable=bad-indentation


import numpy as np


def point_ests(flat):
  '''
  ppint estimates for data offsets. If there are no
  thresholded titres then these will be unbiased how ever
  if there are thresholded titres these will be biased
  '''

  wide = flat.pivot_table(index="lab_assay", values="titre",
                          columns="antigen").values.astype(float)
  lab_offsets = np.nanmean(wide, axis=1)
  gmts = np.nanmean(wide - lab_offsets[:,None], axis=0) +\
    lab_offsets.mean()

  lab_offsets = np.nanmean(wide - gmts[None,:], axis=1)


  I_ass = flat.pivot_table(index="lab_assay", values="assay_type").values[:,0]
  nass = np.unique(I_ass).size
  assay_offsets = np.array([lab_offsets[I_ass==i].mean() for i in range(nass)])


  return gmts, assay_offsets
