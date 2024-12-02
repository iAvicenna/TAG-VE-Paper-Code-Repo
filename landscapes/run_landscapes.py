#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:18:01 2024

@author: avicenna
"""

# pylint: disable=bad-indentation import-error wrong-import-position

import sys
import os
import pickle
from datetime import date

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm

cdir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(f"/{cdir}/../clustering/lib/")
from clustering import censored_titre_clustering_model, sample_observables

sys.path.append(f"{cdir}/lib/")
from landscapes import cone_mixture_model

sys.path.append(f"{cdir}/../")
from common_utils import log, lower, upper, threshold, _HDI, antigens


default_sample_args = {"draws":2000, "tune":2000,
                       "chains":4, "cores":4}

def _sortcentre(yvals):
  c = np.polyfit(range(len(yvals)), yvals, 1)
  return c[0]


def _setup_mixture_params(ag_to_coordinates, titres, ncones, weights_per_cluster,
                         strains, i0):

  params = {}

  mean_titres = np.nanmean(titres.loc[strains,:].values, axis=1)
  if any(np.isnan(x) for x in mean_titres.flatten()):
    mean_titres[np.isnan(mean_titres)] = np.nanmean(titres.loc[strains,:].values,
                                                    axis=1)[np.isnan(mean_titres)]
  params["apex_mixture_mus"] =\
    np.array([ag_to_coordinates[strain] for strain in strains])[None,...]

  params["apex_mixture_mus"] = np.tile(params["apex_mixture_mus"],(1,1,1))

  #nsrgroups x ncones x nnormals
  params["apex_mixture_weights"] = np.array(weights_per_cluster[ncones][i0])[None,...]
  params["height_mixture_weights"] = params["apex_mixture_weights"]

  params["height_mixture_mus"] = mean_titres[None,:]

  params["initvals"] = {}
  params["initvals"]["xcoordinates0"] =  params["apex_mixture_mus"][0,:ncones,0]
  params["initvals"]["ycoordinates0"] =  params["apex_mixture_mus"][0,:ncones,1]
  params["initvals"]["heights0"] = params["height_mixture_mus"][0,:ncones]

  return params


def _sortkey(yvals):
  c = np.polyfit(range(len(yvals)), yvals, 1)
  return c[0]


def _flatten_table(data_table):

  data_table = data_table.rename(columns={"serum_long":"sr_name"})

  flat_table =\
    pd.melt(data_table, id_vars=["lab_code","assay_type", "sr_name"],
            value_vars=["XBB.1.5", "Alpha", "Beta", "Delta", "BA.1", "BA.5"],
            var_name="ag_name", value_name="titre")

  flat_table.loc[:,"lower"] =\
    [lower(x, y + '_' + z.lower()) for
     x,y,z in zip(flat_table.loc[:,"titre"], flat_table.loc[:,"lab_code"],
                  flat_table.loc[:,"assay_type"])]

  flat_table.loc[:,"upper"] =\
    [upper(x, y + '_' + z.lower()) for
     x,y,z in zip(flat_table.loc[:,"titre"], flat_table.loc[:,"lab_code"],
                  flat_table.loc[:,"assay_type"])]

  flat_table.loc[:,"titre"] = flat_table.loc[:,"titre"].apply(log).astype(float)

  return flat_table


def _get_model(titre_table, data_table, nclusters, i0,
               ag_to_coordinates, weights_per_cluster, ncones,
               cluster_likelihoods, centres):

  '''
  this function gets the Bayesian model for the landscapes. In order to
  construct the model, it needs the serum belonging to cluster indexed by i0.
  clusters are determined using cluster_likelihoods. centres (which are the
  cluster centres) are used to order the clusters according to their slope.
  So i0=0 gets the cluster with the biggest slope and so on.
  '''


  flat = _flatten_table(data_table)
  sera = titre_table.index
  labels = np.argmax(cluster_likelihoods, axis=0)
  order = sorted(range(nclusters), key = lambda x: _sortkey(centres[x,:]))


  I = np.argwhere(labels==order[i0]).flatten()
  sorted_likelihoods = np.sort(cluster_likelihoods[:,I], axis=0)
  J = I[np.argwhere((sorted_likelihoods[-2,:]<threshold))].flatten()

  subflat = flat[flat.sr_name.isin(sera[J])]

  if i0==2:
    I = ~subflat.sr_name.isin(['Genève 5']) | ~subflat.ag_name.isin(['Alpha'])
    subflat = subflat[I]
    # this serum with two missing measurements is either misclassified or Alpha
    # is an outlier titre and all the other sera in this group dont have their
    # Alpha titre measured so this is a single, likely unreliable data point
    # for Alpha titres

  if i0!=2:
    params=\
     _setup_mixture_params(ag_to_coordinates,
                          titre_table.loc[sera[J],:].T, ncones,
                          weights_per_cluster, ["Alpha","BA.1","BA.5"], i0)
  else:  #this group does not have measurements fo Alpha
    params=\
     _setup_mixture_params(ag_to_coordinates,
                          titre_table.loc[sera[J],:].T, ncones,
                          weights_per_cluster, ["Beta","BA.1","BA.5"], i0)

  model,meta_data =\
    cone_mixture_model(ncones, subflat, ag_to_coordinates,
                       **params)

  return model, meta_data, params


def _get_cluster_observables(nclusters, outlier_sd_factor=4,
                             centre_sd=1, sr_bias_sd=0.5):
  '''
  gets the cluster likelihoods and centres from clustering folder for the given
  parameters and number of clusters. each serum is assigned a group determined
  by the clustering and then each group is fitted a landscape.
  '''

  with open(f"{cdir}/../clustering/outputs/tables","rb") as fp:
    titre_table, lower_table, upper_table = pickle.load(fp)


  with open(f"{cdir}/../clustering/outputs/kmeans/combined","rb") as fp:
    kmeans_output = pickle.load(fp)

  nsr = titre_table.shape[0]
  centres_kmeans = kmeans_output[nclusters]["cluster_centres"]


  rvec = np.array([1, 1, 1, -1, -1, -1])
  outlier_mean = np.array([np.nanmean(titre_table) for _
                           in range(len(antigens))])
  outlier_sd = 1.6*outlier_sd_factor

  prior_params={
    "mu_sigma":centre_sd,
    "sr_bias_sd":sr_bias_sd
    }

  model, _=\
    censored_titre_clustering_model(titre_table, nclusters,
                                    lower=lower_table.values,
                                    upper=upper_table.values, rvec=rvec,
                                    mu_ests = centres_kmeans,
                                    outlier_sd=outlier_sd,
                                    outlier_mean=outlier_mean,
                                    prior_params=prior_params)

  with open(f"{cdir}/../clustering/outputs/bayesian/idata{nclusters}","rb") as fp:
    idata, _ = pickle.load(fp)

  assert idata["posterior"]["mu1"]["cluster"].size == len(model.coords["cluster"])

  sample_observables(idata, np.array([1, 1, 1, -1, -1, -1]), model)

  with model:
    pm.compute_log_likelihood(idata)

  N = nclusters + 1*int(nclusters>1) # when nclusters>1 there is an outliers cluster

  with np.errstate(divide="ignore", invalid="ignore"):
    summary = az.summary(idata, group="posterior",
                         var_names=["cluster_likelihoods"],
                         hdi_prob=_HDI).iloc[:,0].values

    cluster_likelihoods =  np.reshape(summary, (N,nsr))

    summary = az.summary(idata, group="predictions",
                         var_names=["mu"],
                         hdi_prob=_HDI).iloc[:,0].values

    mu =  np.reshape(summary, (nclusters, len(antigens)))

  return cluster_likelihoods, mu


def _main(sample_args=None, load=False, ncones_range=None,
          groups_range=None):

  if ncones_range is None:
    ncones_range = [1,2]
  if groups_range is None:
    groups_range = [0,1,2]

  if sample_args is None:
    sample_args = {}

  sample_args = dict(default_sample_args, **sample_args)

  with open(f"{cdir}/../clustering/outputs/tables","rb") as fp:
    titre_table,_,_ = pickle.load(fp)

  data_table = pd.read_csv(f"{cdir}/../clustering/outputs/subsetted_table.csv",
                           header=0, index_col=None)

  with open(f"{cdir}/data/ag_data","rb") as fp:
    ag_data = pickle.load(fp)

  base_map = {}
  base_map["ag_coordinates"] = [x[0] for x in ag_data.values()]
  base_map["ag_colours"] = [x[1] for x in ag_data.values()]
  base_map["ag_names"] = list(ag_data.keys())
  ag_to_coordinates = {ag:ag_data[ag][0] for ag in ag_data}

  weights_per_cluster = {1:[[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]]],
                          2:[[[1, 0, 0],[0, 1, 1]],
                            [[1, 0, 0],[0, 1, 1]],
                            [[0, 1, 0], [1, 0, 1]]]
                            }
  nclusters = 3

  cluster_likelihoods, centres =\
    _get_cluster_observables(nclusters)


  for i0 in groups_range:
    for ncones in ncones_range:

      model, meta_data, params=\
       _get_model(titre_table.copy(), data_table.copy(), nclusters,  i0,
                  ag_to_coordinates.copy(), weights_per_cluster.copy(),  ncones,
                  cluster_likelihoods.copy(), centres.copy())

      # you cant pickle models, so saving necessary input to rebuild the model
      # for posterior predictive sampling etc
      meta_data["model_args"] = {"titre_table":titre_table,
                                 "data_table":data_table,
                                 "ag_to_coordinates":ag_to_coordinates,
                                 "weights_per_cluster": weights_per_cluster,
                                 "cluster_likelihoods": cluster_likelihoods,
                                 "centres":centres}
      meta_data["base_map"] = base_map

      if load:
        with open(f"{cdir}/outputs/group{i0}_ncones{ncones}","rb") as fp:
          idata, meta_data, _ = pickle.load(fp)

      else:
        with model:
          idata = pm.sample(**sample_args, initvals=meta_data["initvals"])


        divergences = np.array(idata.sample_stats['diverging']).sum()
        rhat = np.argwhere(az.summary(idata).iloc[:,-1]>1.01).flatten().size

        with open(f"{cdir}/outputs/logs/{i0}_{ncones}", "w", encoding="utf-8") as fp:
          fp.write(f"{date.today()}\n")
          fp.write(f"divergences: {divergences}\n")
          fp.write(f"rhats: {rhat}\n")
          fp.write(f"{params}")
          fp.write("\n")

        with open(f"{cdir}/outputs/group{i0}_ncones{ncones}","wb") as fp:
          pickle.dump([idata, meta_data, params], fp)



if __name__ == "__main__":

  _main(groups_range=[1,2], ncones_range=[1,2])
