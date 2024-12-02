#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:15:46 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error

# pylint: disable=unsubscriptable-object
# for some reason pylint does not know pymc distributions are subscriptable

# pylint: disable=protected-access
# I am accesing underscored members I have created, they are not protected
# but rather _ is used to indicate they are uncentred parameters

import sys
import os
from functools import partial

cdir = os.path.dirname(os.path.realpath(__file__))

import pymc as pm
import pandas as pd
import numpy as np

sys.path.append(cdir)

from _utils import point_ests

default_prior_params = {
  "noise_sd_mu":1,
  "noise_sd_sd":0.5,
  "gmt_sigma":1,
  "assay_offsets_mu_sigma":1,
  "dataset_offsets_sigma":2,
  "nu_mu":5,
  "nu_sigma":2
  }

_REQ_KEYS = set(["assay_type", "lab_code", "antigen",
                 "titre", "lower", "upper"])
_LVL_KEYS = set(["assay_type", "lab_code", "antigen"])
assert _LVL_KEYS.issubset(_REQ_KEYS),\
  "module level error, _LVL_KEYS must be subset of _REQ_KEYS"


def gmt_bayesian_model(flat: pd.DataFrame, prior_params:dict=None,
                       use_t=True):

  '''
  flat: dataset to be used, see _process_level_sets for requirements
  prior_params: parameters to be used in priors, see default_prior_params
  use_t: whether to use student T or normal for noise
  '''

  #drop unobserved
  flat = flat.copy()[~flat.titre.isin([np.nan])]

  if prior_params is None:
    prior_params = {}
  prior_params = dict(default_prior_params, **prior_params)

  # converts values such as lab_code, assay_type etc into numerical values
  # and returns a list of corresponding level sets for each such column
  # it also extends the flat by some more columns which are used in the model
  # for indexing, coordinate purposes etc
  level_sets, flat = _process_level_sets(flat.copy())

  meta = {}
  meta["level_sets"] = level_sets
  meta["flat"] = flat
  meta["prior_params"] = prior_params

  #indexer that maps each lab_assay to its corresponding assay within level_sets
  I = flat.pivot_table(index="lab_assay", values="assay_type").values[:,0].\
    astype(int)

  # observation coordinates, used for giving a human readable name to each
  # observation
  obs_coords = [level_sets["lab_assay_antigen"][i] for i in
                flat.loc[:,"lab_assay_antigen"].values]

  coords = dict(level_sets, **{"obs_name":obs_coords})

  meta["coords"] = coords

  # the data is centred so that posterior predictive plots are more
  # interpretable. the parameter posterior distributions whose names start
  # with _ can be transformed back using the sample_centred_observables
  # function
  gmts_pt_est, assay_offsets_pt_est = point_ests(flat)

  meta["pt_ests"] = {
    "gmts":gmts_pt_est,
    "assay_offsets":assay_offsets_pt_est,
    "assay_offsets_sd": assay_offsets_pt_est,
    }

  with pm.Model(coords=coords) as model:

    # constants to be recorded some of which are used later in
    # sample_centred_observables and some others used for resetting
    # data etc when doing leave one out cross-validation for data with
    # high pareto values

    gmts_pt_est = pm.Data("gmts_pt_est", gmts_pt_est, dims="antigen")
    data = pm.Data("data", flat.loc[:,"titre"].astype(float),
                   dims="obs_name")

    lower = pm.Data("lower", flat.loc[:,"lower"].astype(float),
                    dims="obs_name")
    upper = pm.Data("upper", flat.loc[:,"upper"].astype(float),
                    dims="obs_name")
    I_ag = pm.Data("I_ag", flat.loc[:,"antigen"].values.astype(int),
                   dims="obs_name")
    I_lass = pm.Data("I_lass", flat.loc[:,"lab_assay"].values.astype(int),
                    dims="obs_name")

    #priors
    gmts = pm.Normal("_gmts",
                     mu = gmts_pt_est,
                     sigma = prior_params["gmt_sigma"],
                     dims="antigen",
                     shape= len(level_sets["antigen"]))

    noise_sd = pm.InverseGamma("noise_sd",
                               mu=prior_params["noise_sd_mu"],
                               sigma=prior_params["noise_sd_sd"],
                               dims="antigen",
                               shape=len(level_sets["antigen"])
                               )


    if use_t:
      nu = pm.InverseGamma("nu", mu=prior_params["nu_mu"],
                           sigma=prior_params["nu_sigma"])

    assay_offsets_mu =\
      pm.Normal("_assay_offsets_mu", assay_offsets_pt_est,
                sigma=prior_params["assay_offsets_mu_sigma"],
                shape=(len(level_sets["assay_type"])),
                dims="assay_type")


    dataset_offsets =\
      pm.Normal("_dataset_offsets",
                mu=assay_offsets_mu[I],
                sigma=prior_params["dataset_offsets_sigma"],
                shape=(len(level_sets["lab_assay"])),
                dims="lab_assay"
                ) #offset for every lab+assay combination


    #transformed priors
    fitted_titres =  gmts[I_ag] + dataset_offsets[I_lass]

    if use_t:
      dist = pm.StudentT.dist(nu=nu, mu=fitted_titres,
                              sigma=noise_sd[I_ag])
    else:
      dist = pm.Normal.dist(mu=fitted_titres,
                            sigma=noise_sd[I_ag])

    pm.Censored("obs", dist, lower=lower, upper=upper,
                observed=data, dims="obs_name")


  return model, meta



def sample_centred_observables(model, idata):

  '''
  apply inverse data centring transformation and then centre the
  posterior distributions
  '''

  I_lass = idata["constant_data"]["I_lass"].to_numpy()

  var_names = []

  with model:

    model.coords["antigen2"] = model.coords["antigen"] # used for pairwise fold-drops
    model.coords["assay_type2"] = model.coords["assay_type"] # used for pairwise assay offset dif

    gmts = model._gmts + model._dataset_offsets.mean()

    gmts=\
      pm.Deterministic("gmts_centred", gmts, dims="antigen")

    assay_offsets_centred = model._assay_offsets_mu -\
      model._assay_offsets_mu.mean()
    pm.Deterministic("assay_offsets_mu_centred", assay_offsets_centred,
                     dims="assay_type")

    lab_titres_wo_offset = idata["observed_data"]["obs"].to_numpy() -\
      model._dataset_offsets[I_lass]

    pm.Deterministic("lab_titres_wo_offset", lab_titres_wo_offset,
                     dims="obs_name")

    lab_assay_codes = model.coords["lab_assay"]
    antigens = model.coords["antigen"]
    obs_names = model.coords["obs_name"]


    for code0 in lab_assay_codes:

      coords0 = [x for x in obs_names if code0 in x]
      ags0 = [x.split('_')[-1] for x in coords0]
      I0 = [antigens.index(i) for i in ags0]
      _gmts = gmts[I0]

      I1 = [obs_names.index(i) for i in coords0]
      titres0 = lab_titres_wo_offset[I1]


      pm.Deterministic(f"gmt_{code0}_dif", _gmts - titres0)
      var_names.append(f"gmt_{code0}_dif")

      for code1 in lab_assay_codes:

        if code0==code1:
          continue

        coords0 = [x for x in obs_names if code0 in x]
        coords1 = [x for x in obs_names if code1 in x]

        ags0 = [x.split('_')[-1] for x in coords0]
        ags1 = [x.split('_')[-1] for x in coords1]

        common_ags = [ag for ag in antigens if ag in ags0 and ag in ags1]

        coords0 = [x for x,ag in zip(coords0,ags0) if ag in common_ags]
        coords1 = [x for x,ag in zip(coords1,ags1) if ag in common_ags]

        I0 = [obs_names.index(i) for i in coords0]
        I1 = [obs_names.index(i) for i in coords1]

        titres0 = lab_titres_wo_offset[I0]
        titres1 = lab_titres_wo_offset[I1]


        pm.Deterministic(f"{code0}_{code1}_dif", titres0 - titres1)
        var_names.append(f"{code0}_{code1}_dif")


    dataset_offsets_centred = model._dataset_offsets -\
      model._dataset_offsets.mean()
    pm.Deterministic("dataset_offsets_centred", dataset_offsets_centred,
                     dims="lab_assay")

    pm.Deterministic("pairwise_fold_drops", model._gmts[:, None] -
                     model._gmts[None, :], dims=["antigen", "antigen2"])

    pm.Deterministic("pairwise_assay_mu_differences",
                     model._assay_offsets_mu[None,:] -
                     model._assay_offsets_mu[:,None],
                     dims=["assay_type","assay_type2"])


    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   predictions=True,
                                   var_names=["gmts_centred",
                                              "assay_offsets_mu_centred",
                                              "dataset_offsets_centred",
                                              "pairwise_fold_drops",
                                              "pairwise_assay_mu_differences",
                                              "lab_titres_wo_offset"
                                              ]+var_names)
    return idata


def _process_level_sets(flat):

  '''
  Given a flat table with columns containing those in _LVL_KEYs,
  converts column values to integers and keeps record of level_sets
  for these columns. Also extends flat by two new columns lab_assay,
  lab_assay_antigen which are level sets of combinations of the corresponding
  columns: lab, assay, antigen
  '''

  if not all(x in flat.columns for x in _REQ_KEYS):
    missing = [x for x in _REQ_KEYS if x not in flat.columns]
    raise ValueError(f"following columns must exist in the table: {missing}")

  flat.loc[:,"lab_assay"] = [f"{x}_{y.lower()}" for x,y in
                             zip(flat.loc[:,"lab_code"],
                                 flat.loc[:,"assay_type"])]
  flat.loc[:,"lab_assay_antigen"] = [f"{x}_{y.lower()}_{z}" for x,y,z in
                                     zip(flat.loc[:,"lab_code"],
                                         flat.loc[:,"assay_type"],
                                         flat.loc[:,"antigen"])]

  level_sets = {}
  new_flat = flat.copy()

  def _level_index(levels, x):
    return levels.index(x)

  for key in _LVL_KEYS.union(set(["lab_assay","lab_assay_antigen"])):
    level_sets[key] = sorted(list(set(flat.loc[:, key].values)))

    _index_fun = partial(_level_index, level_sets[key])
    new_flat.loc[:,key] = flat.loc[:,key].map(_index_fun)


  return level_sets, new_flat
