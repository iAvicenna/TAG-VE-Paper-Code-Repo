#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:33:13 2024

@author: avicenna
"""

# pylint: disable=bad-indentation


import pymc as pm
import arviz as az
import pytensor.tensor as pt
import numpy as np

from xarray import DataArray


def gmt_bayesian_model_log_likelihood(gmts, dataset_offsets, noise_sd,
                                      I_ag, I_ass, lower, upper, data,
                                      nu=None):

  '''
  this returns only the likelihood(data|param) without any priors
  for the model gmt_bayesian_model in models.py

  it is used for leave-one-out cross-validation
  '''

  with pm.Model() as like:

    #transformed priors
    row_offsets = dataset_offsets[...,I_ass]

    fitted_titres = gmts[...,I_ag] + row_offsets

    if nu is None:
      dist = pm.Normal.dist(mu=np.array(fitted_titres),
                            sigma=np.array(noise_sd)[...,I_ag])
    else:
      dist = pm.StudentT.dist(nu=np.array(nu), mu=np.array(fitted_titres),
                              sigma=np.array(noise_sd)[...,I_ag])


    pm.Censored("obs", dist, lower=np.array(lower)[0],
                upper=np.array(upper)[0],
                observed=np.array(data)[:,None,None])

  value = pt.tensor("value", dtype=float, shape=(1,))
  logp_fun = pm.compile_pymc([value], pm.logp(like.obs, value))

  return logp_fun


class PyMCLOOWrapper(az.PyMCSamplingWrapper):
  '''
  When pareto k values for loo is low, exact
  leave one out cross-validation is required.

  This is a wrapper class for doing that. modified from
  https://python.arviz.org/en/stable/user_guide/pymc_refitting.html
  '''

  def sample(self, modified_observed_data):

    with self.model:

      n__i = len(modified_observed_data["lower"])
      self.model.set_dim("obs_name", n__i, coord_values=np.arange(n__i))
      pm.set_data(modified_observed_data)


      idata = pm.sample(
          **self.sample_kwargs,
          return_inferencedata=True,
          idata_kwargs={"log_likelihood": True}
      )


    return idata


  def log_likelihood__i(self, excluded_obs, idata__i):

    # get posterior trained on non excluded observables
    post = idata__i.posterior

    # construct the function data -> log likelihood(data|param)
    if "nu" in excluded_obs:
      nu = excluded_obs["nu"]
    else:
      nu = None

    logp_fun =\
      gmt_bayesian_model_log_likelihood(post["_gmts"],
                                        post["_dataset_offsets"],
                                        post["noise_sd"],
                                        excluded_obs["I_ag"],
                                        excluded_obs["I_lass"],
                                        excluded_obs["lower"],
                                        excluded_obs["upper"],
                                        excluded_obs["data"],
                                        nu=nu)


    # return log likelihood (excluded obs | params trained on non excluded)
    return DataArray(logp_fun(value=excluded_obs["data"]))


  def sel_observations(self, idx):

    '''
    create a dictionary mapping pm.Data names to excluded and non-excluded
    values so they can be set in sample
    '''

    lower = self.idata_orig["constant_data"]["lower"]
    upper = self.idata_orig["constant_data"]["upper"]
    I_ag = self.idata_orig["constant_data"]["I_ag"]
    I_ass = self.idata_orig["constant_data"]["I_lass"]
    data = self.idata_orig["observed_data"]["obs"]


    mask = np.isin(np.arange(lower.size), idx)
    data_dict = {"lower": lower, "upper": upper, "I_ag":I_ag,
                 "I_lass":I_ass, "data": data,
                 }

    data__i = {key: value.values[~mask] for key, value in data_dict.items()}
    data_ex = {key: value.isel({"obs_name":idx}) for key, value in data_dict.items()}


    return data__i, data_ex
