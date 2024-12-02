#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:31:08 2024

@author: avicenna
"""
# pylint: disable=bad-indentation

# pylint: disable=unsubscriptable-object
# pylint for some reason thinks pymc random variables are unsubscriptable
# which is wrong

import warnings
import sys
import os

cdir = os.path.dirname(os.path.realpath(__file__))

import pymc as pm
import numpy as np
import pytensor.tensor as pt

sys.path.append(cdir)

from _utils import split_data_by_missing_pattern
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

trans = pm.distributions.transforms.ordered

#some of these params, if not supplied will be determined based on data
default_prior_params = {"sr_bias_sd":None,
                        "noise_mu":0.5,
                        "noise_sigma":0.25,
                        "mu_sigma":None,
                        "weight_concentration":20,
                        }

# dist and logp below are used to make a batched collection of censored
# uv normals into a multivariate distribution so they are sampled together
# and are not mixed coordinate-wise. If this is not used and normals are
# batched instead results will likely look much worse

def dist(mu, sigma, lower, upper, size=None):

  '''
  a distribution to define collection of univariate censored normals as
  a multivariate distribution
  '''

  mu = pt.as_tensor_variable(mu)

  return\
    pm.Censored.dist(pm.Normal.dist(mu, sigma),
                     lower=lower, upper=upper)


def logp(value, *params):
  '''
  log probability which is used to define a collection of UV distributions
  as a multivariate by summing the logp for each UV distribution
  '''
  # Recreate the RV
  rv = dist(*params)

  # Request logp
  logp_val = pm.logp(rv, value)
  # Sum last axis

  return logp_val.sum(axis=-1)



def censored_titre_clustering_model(data, n_clusters, rvec=None,
                                    mu_ests=None, outlier_mean=None,
                                    outlier_sd=None, outlier_sd_factor=6,
                                    lower=None, upper=None, prior_params=None):

  '''
  this is a bayesian model for clustering via Mixture of Censored Normals.
  for a description of the model see the paper.

  data should be a numpy array with columns depicting coordinates rows
  depicting data. It is allowed to have unmeasured values given by np.nan.
  The data is grouped into patterns of missing data via the
  split_data_by_missing_pattern function.

  lower and upper denote thresholded values and if not supplied assumed to
  be -np.inf and np.inf. They should have the same shape as data.

  if outlier_sd is supplied, that is used for defining the fixed standard
  deviation of outlier cluster. If not then it is outlier_sd_factor x
  data standard deviation.

  rvec determines how to construct the Rank coordinate. If
  rvec = [1, 0 ,... ,0] then  it is equal to first coordinate of the
  data. If rvec = [1, -1, 0, 0, ..., 0] then it is first - second, etc.
  It is given by Rank = coordinates * rvec. Rank is the coordinate used
  to order clusters via the pm.distributions.transforms.ordered transformation.

  mu_ests should be a list of coordinates with length equal to number of
  clusters if given. It gives a prior guess on the cluster centres. Otherwise
  it is determined by data. In the paper we use initial guesses by kmeans.
  '''


  if prior_params is None:
    prior_params = {}

  prior_params = dict(default_prior_params, **prior_params)

  data_bias = np.nanmean(data, axis=1)[:,None]
  data = data.copy() - data_bias
  lower = lower.copy() - data_bias
  upper = upper.copy() - data_bias

  n_dims = data.shape[1]

  alpha0 = [prior_params["weight_concentration"] for _ in range(n_clusters)]
  if n_clusters>1:
    alpha0 += [prior_params["weight_concentration"]/10]

  if outlier_mean is None:
    data_mean = np.nanmean(data, axis=0)
    outlier_mean = data_mean
  else:
    assert outlier_mean.shape == (n_dims,)
    outlier_mean = outlier_mean.copy() - np.mean(outlier_mean)

  if outlier_sd is None:
    data_sd = np.mean(np.nanstd(data, axis=0))
    outlier_sd = outlier_sd_factor*data_sd

  if rvec is None:
    dvec = [1/i for i in range(1,n_dims)]
    rvec = np.array([-np.sum(dvec)] + [dvec[i] for i in range(n_dims-1)])*1/(n_dims-1)

  grouped_data = split_data_by_missing_pattern(data)

  if lower is None:
    lower = -np.inf*np.ones(data.shape)
  else:
    assert isinstance(lower,(float,int)) or lower.shape == data.shape

    if isinstance(lower, (float,int)):
      lower = lower*np.ones(data.shape)

  if upper is None:
    upper = np.inf*np.ones(data.shape)
  else:
    assert isinstance(upper, (float,int)) or upper.shape == data.shape

    if isinstance(upper, (float,int)):
      upper = upper*np.ones(data.shape)

  if mu_ests is None:
    mu_ests = np.tile(np.nanmean(data,axis=0)[None,:],
                                    (n_clusters,1))
    mu_ests += np.random.normal(0, 0.5,
                                mu_ests.shape)
  else:
    assert mu_ests.shape == (n_clusters,n_dims)

  mu_ests = mu_ests - np.nanmean(mu_ests, axis=1)[:,None]

  mu1_init =   mu_ests[:,1:]
  mu_ests = [mu_ests[:,0], mu_ests[:,1:]]

  if prior_params["mu_sigma"] is None:
    prior_params["mu_sigma"] = 0.5*np.nanstd(data, axis=0)
  else:
    if isinstance(prior_params["mu_sigma"], (int,float)):
      prior_params["mu_sigma"] =\
        np.array([prior_params["mu_sigma"] for _ in range(n_dims)])

  if prior_params["sr_bias_sd"] is None:
    prior_params["sr_bias_sd"] = 1/3*np.mean(np.nanmax(data,axis=0)\
                                             - np.nanmin(data,axis=0))

  if n_clusters>1:
    R_init = mu_ests[0] * rvec[0] + mu_ests[1] @ rvec[1:]
  else:
    R_init = mu_ests[0] * rvec[0]

  R_data = data @ rvec[:,None]

  if mu1_init is not None:
    mu1_init = mu1_init[np.argsort(R_init),:]
  R_init = np.sort(R_init)


  coords={"cluster": np.arange(n_clusters),
          "cluster_woutlier": [str(x) for x in np.arange(n_clusters)] +\
            ["outlier"],
          "obs": data.index,
          "coord":data.columns,
          "coord[1:]":data.columns[1:],
          }

  meta = {}
  meta["R_data"] = R_data


  initvals={"rank":R_init, "mu1":mu1_init,
            "w":alpha0,
            "sigma":1}

  meta["initvals"] = initvals
  meta["grouped_data"] = grouped_data
  meta["outlier_mean"] = outlier_mean
  meta["rvec"] = rvec

  with pm.Model(coords=coords) as model:

      # allow smaller clusters with somewhat of a sparse alpha
      # and about 1/10 weight for the outliers cluster
      w = pm.Dirichlet("w", alpha0)

      R =\
        pm.Normal("rank", mu=R_init,
                  sigma=np.sqrt(np.sum(np.abs(rvec)*
                                       prior_params["mu_sigma"]**2)),
                  shape=n_clusters,
                  transform=pm.distributions.transforms.ordered,
                  dims=("cluster",))

      mu1 = pm.Normal("mu1",
                      mu=mu1_init,
                      sigma=prior_params["mu_sigma"][1:],
                      shape=(n_clusters, n_dims-1),
                      dims=("cluster","coord[1:]"))

      if n_clusters>1:
        mu0 = (R - mu1@rvec[1:])/rvec[0]

      else:
        mu0 = R/rvec[0]


      mu = pt.concatenate([mu0[:,None], mu1], axis=1)

      if n_clusters>1:
        mu = pt.concatenate([mu, outlier_mean[None,:]], axis=0)

      sigma = pm.InverseGamma('sigma', mu=prior_params["noise_mu"],
                              sigma=prior_params["noise_sigma"])
      sigmas = [sigma for _ in range(n_clusters)]

      if n_clusters>1:
        sigmas += [outlier_sd]

      sigmas = pt.stack(sigmas)
      serum_biases = pm.Normal("b", 0, prior_params["sr_bias_sd"],
                               dims=("obs",))

      # Define the mixture per group
      sample_cluster_likelihoods(w, grouped_data, lower, upper, n_clusters,
                                 mu, sigmas, serum_biases, model)

      for ind_group,group in enumerate(grouped_data):

        _coords = np.argwhere(group[0]).flatten()
        _index = group[1]
        _data = group[2]
        _serum_biases = serum_biases[_index]

        if n_clusters>1:
          components =\
            [pm.CustomDist.dist(mu[i][_coords][None,:] + _serum_biases[:,None],
                                sigmas[i],
                                lower[np.ix_(_index,_coords)],
                                upper[np.ix_(_index,_coords)],
                                logp=logp, dist=dist,
                                ndim_supp=1) for i in range(n_clusters+1)]

          pm.Mixture(f"mix{ind_group}", w=w,
                     comp_dists=components,
                     observed=_data)

        else:
          components=\
          pm.CustomDist(f"mix{ind_group}",
                        mu[0][_coords][None,:] + _serum_biases[:,None],
                        sigma,
                        lower[np.ix_(_index,_coords)],
                        upper[np.ix_(_index,_coords)],
                        logp=logp, dist=dist,
                        ndim_supp=1, observed=_data)


  return model, meta



def sample_cluster_likelihoods(w, grouped_data, lower, upper, n_clusters, mu,
                               sigmas, serum_biases, model):

  '''
  since mixture models are effectively marginalizing cluster indices,
  this function is used to recover them back by computing the probability
  that each data belongs to a given cluster.
  '''

  if n_clusters==1:
    N = 1
  else:
    N = n_clusters + 1

  for ind_group,group in enumerate(grouped_data):
    _coords = np.argwhere(group[0]).flatten()
    _index = group[1]
    _data = group[2]
    _serum_biases = serum_biases[_index]

    components =\
      [pm.CustomDist.dist(mu[i][_coords][None,:] + _serum_biases[:,None],
                          sigmas[i],
                          lower[np.ix_(_index,_coords)],
                          upper[np.ix_(_index,_coords)],
                          logp=logp, dist=dist,
                          ndim_supp=1) for i in range(N)]


    logp_safe_components = model.replace_rvs_by_values(components)
    _log_ps =\
      pt.stack([pm.logp(logp_safe_components[i], _data)
                 for i in range(N)])

    if ind_group == 0:
      log_ps = _log_ps
    else:
      log_ps = pt.concatenate([log_ps, _log_ps],axis=1)

  # these two lines here make sure that cluster likelihoods are
  # arranged in the same order as the titre table
  I = [x for y in grouped_data for x in y[1]]
  J = [I.index(x) for x in range(len(I))]

  p = pm.math.exp(log_ps[:,J])

  normalization = (w[:,None]*p).sum(axis=0)

  cluster_likelihoods=\
    pm.Deterministic("cluster_likelihoods", w[:,None]/normalization[None,:]*p)

  cohesion = pt.max(cluster_likelihoods, axis=0)

  if n_clusters>1:
    seperation = pt.sort(cluster_likelihoods, axis=0)[-2,:]
    pm.Deterministic("silhoutte_scores", cohesion-seperation)
  else:
    pm.Deterministic("silhoutte_scores", cohesion)


def sample_observables(idata, rvec, model):

  '''
  used to sample_observables after posterior sampling is complete.
  it samples
    mu: cluster centres including outlier cluster (which is fixed)
    folddrops: folddrops of each cluster centre to the first coordinate
    folddrop differences: pairwise difference between computed folddrops
      which is a measure of how different each cluster center is
  '''

  n_clusters = len(model.coords["cluster"])
  var_names = ["mu", "folddrops"]

  with model:
    R = model.rank
    mu1 = model.mu1

    if n_clusters>1:
      mu0 = (R - mu1@rvec[1:])/rvec[0]
    else:
      mu0 = R/rvec[0]

    mu = pm.Deterministic("mu",
                          pt.concatenate([mu0[:,None], mu1], axis=1),
                          dims=["cluster","coord"])

    fd = pm.Deterministic("folddrops", mu - mu[:,0:1],
                          dims=["cluster","coord"])

    if n_clusters>1:
      pm.Deterministic("folddrop_differences",
                        pt.transpose(fd[:,:,None] - fd.T[None,:,:],
                                    axes=[0, 2, 1])) #nclusters x nclusters x ndims
      var_names.append("folddrop_differences")



    pm.sample_posterior_predictive(idata, extend_inferencedata=True,
                                   predictions=True,
                                   var_names=var_names)


def kmeans_missing(X, N, max_iter=10, normalize=False, verbose=True):
    """
    Perform spectral clustering on data with missing values.
    If there are no missing values it does normal spectral clustering

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      N: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.
      seed: seed for spectral clustering
      nneighbours: number of nearest neighbours to construct the graph

    Returns:
      For each of the attempted number of clusters it returns:
      labels, silhoutte scores, within cluster mse, imputed Xs, cluster centres

     original author: ali_m in stackoverflow:
       https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data

     modifications: spectral clustering is used here instead of k-means. therefore
     instead of resupplying initial conditions to get consistent labels as
     in the original code, a label matching algorithm based on
     linear_sum_assignment is used to match labels from one step to next.
    """

    if normalize:
      shift = np.tile(np.nanmean(X, axis=1)[:,None], (1, X.shape[1]))
    else:
      shift = np.zeros(X.shape)

    X = X.copy() - shift

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    if any(np.isnan(x) for x in mu.flatten()) and verbose:
      raise ValueError("There is a whole column of missing values. "
                       "Drop that column.")


    output = {}

    if np.where(missing)[0].size == 0:
        max_iter=1


    for nclusters in range(1,N+1):

        X_hat = np.where(missing, mu, X)

        prev_labels = None
        converged = False
        if verbose:
          print(f"n clusters: {nclusters}")

        for i in range(max_iter):

          cls = KMeans(n_clusters=nclusters, n_init="auto")

          # perform clustering on the filled-in data
          with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            labels = cls.fit_predict(X_hat)

          if nclusters>1 and prev_labels is not None:
            labels = match_labels(prev_labels.copy(), labels.copy())

          cluster_centres_nonshifted =\
            np.array([(X_hat)[labels == i].mean(axis=0)
                      if np.count_nonzero(labels == i)>0
                      else X_hat.mean(axis=0)
                      for i in range(nclusters)])

          cluster_centres = np.array([(X_hat+shift)[labels == i].mean(axis=0)
                                      if np.count_nonzero(labels == i)>0
                                      else (X_hat+shift).mean(axis=0)
                                      for i in range(nclusters)])

          clusterwise_errs = np.zeros((nclusters,))

          for point, label in zip(X_hat, labels):
            clusterwise_errs[label] +=\
              np.sum([x for x in np.square(point -
                                           cluster_centres_nonshifted[label]).flatten()
                      if not np.isnan(x)])

          mean_err = np.sum(clusterwise_errs)/X.shape[0]
          if nclusters>1:
            score = silhouette_score(X_hat, labels)
          else:
            score = np.nan

          # fill in the missing values based on their cluster centroids
          X_hat[missing] = cluster_centres_nonshifted[labels,:][missing]

          # when the labels have stopped changing then we have converged
          # or when there is no missing titres
          if (i > 0 and np.all(labels == prev_labels))\
            or np.count_nonzero(missing)==0:

            converged = True
            break

          prev_labels = labels

        #report convergence if there are missing titres
        if not converged and np.count_nonzero(missing)!=0 and verbose:
          print(f"Warning: convergence not achieved at iteration {i}.")
        elif  np.count_nonzero(missing)!=0 and verbose:
          print(f"Convergence achieved at iteration {i}")

        output[nclusters] = {}
        output[nclusters]["err"] = mean_err
        output[nclusters]["score"] = score
        output[nclusters]["labels"] = labels
        output[nclusters]["cluster_centres"] = cluster_centres
        output[nclusters]["imputed_data"] = X_hat + shift


    return output


def match_labels(labels_old, labels_new, max_iter=100, verbose=False):

    '''
    reassign labels in labels_new as best as possible to match those in
    labels_old. it is not a procrustes however in that trying to match
    [0,0,1,1,1,1] to [0,0,0,0,1,1] could produce unreliable results.
    It does allow for unequal number of elements in each label list.
    used in kmeans_missing
    '''

    levels_old = sorted(list(set(labels_old)))
    levels_new = sorted(list(set(labels_new)))

    n0 = len(levels_old)
    n1 = len(levels_new)

    cost_matrix = np.zeros((n0, n1))
    converged = False

    for ind2 in range(max_iter):

      # cost matrix is similar to confusion matrix but can handle
      # variable number of labels in each list
      for ind0 in range(n0):
        for  ind1 in range(n1):

          vals1 = [lind for lind,label1 in enumerate(labels_old)
                   if label1==levels_old[ind0]]
          vals2 = [lind for lind,label2 in enumerate(labels_new)
                   if label2==levels_new[ind1]]

          cost_matrix[ind0,ind1] = len(set(vals1).difference(vals2)) +\
            len(set(vals2).difference(vals1))

      I = linear_sum_assignment(cost_matrix.T)

      assignment = dict(zip(I[0],I[1]))
      prev_labels = labels_new.copy()
      labels_new = np.array([levels_old[assignment[levels_new.index(x)]]
                             if levels_new.index(x) in assignment else
                             x for x in labels_new])

      levels_old = sorted(list(set(labels_old)))
      levels_new = sorted(list(set(labels_new)))
      n0 = len(levels_old)
      n1 = len(levels_new)

      if all(x==y for x,y in zip(labels_new, prev_labels)):
        converged = True

      if all(x==y for x,y in zip(labels_new, prev_labels)) or ind2==max_iter:
        break

    if not converged and verbose:
      print(f"Warning: Matching converged at iteration {ind2}")
    elif verbose:
      print(f"Matching converged at iteration {ind2}")

    return labels_new



def censored_titre_clustering_likelihood(data, n_clusters, idata,
                                         rvec=None, outlier_mean=None,
                                         outlier_sd=None, outlier_sd_factor=6,
                                         lower=None, upper=None):
  '''
  this returns the prior probability free likelihood of the
  model censored_titre_clustering_model above.

  it is used in determining the likelihood of the model for a given number
  of clusters and fixed paramaters.
  '''


  w = np.array(idata["posterior"]["w"])
  R = np.array(idata["posterior"]["rank"])
  mu1 = np.array(idata["posterior"]["mu1"])
  sigma = np.array(idata["posterior"]["sigma"])
  serum_biases = np.array(idata["posterior"]["b"])

  data_bias = np.nanmean(data, axis=1)[:,None]
  data = data.copy() - data_bias
  lower = lower.copy() - data_bias
  upper = upper.copy() - data_bias

  n_dims = data.shape[1]

  if outlier_mean is None:
    data_mean = np.nanmean(data, axis=0)
    outlier_mean = data_mean
  else:
    assert outlier_mean.shape == (n_dims,)
    outlier_mean = outlier_mean.copy() - np.mean(outlier_mean)

  if outlier_sd is None:
    data_sd = np.mean(np.nanstd(data, axis=0))
    outlier_sd = outlier_sd_factor*data_sd

  if rvec is None:
    dvec = [1/i for i in range(1,n_dims)]
    rvec = np.array([-np.sum(dvec)] + [dvec[i] for i in range(n_dims-1)])*1/(n_dims-1)

  grouped_data = split_data_by_missing_pattern(data)

  if lower is None:
    lower = -np.inf*np.ones(data.shape)
  else:
    assert isinstance(lower,(float,int)) or lower.shape == data.shape

    if isinstance(lower, (float,int)):
      lower = lower*np.ones(data.shape)

  if upper is None:
    upper = np.inf*np.ones(data.shape)
  else:
    assert isinstance(upper, (float,int)) or upper.shape == data.shape

    if isinstance(upper, (float,int)):
      upper = upper*np.ones(data.shape)


  logps = []
  ns = mu1.shape[0]*mu1.shape[1]
  serum_biases = pt.reshape(serum_biases, (ns, serum_biases.shape[-1]))
  sigma = np.reshape(sigma, (ns,))
  w = np.reshape(w, (ns, w.shape[-1]))


  with pm.Model() as like:

      # allow smaller clusters with somewhat of a sparse alpha
      # and about 1/10 weight for the outliers cluster


      if n_clusters>1:
        mu0 = (R - mu1@rvec[1:])/rvec[0]

      else:
        mu0 = R/rvec[0]

      mu = pt.concatenate([mu0[...,None], mu1], axis=-1)

      if n_clusters>1:
        outlier_mean = np.tile(outlier_mean[None,None,None,:],
                               (mu0.shape[0], mu0.shape[1], 1, 1))
        mu = pt.concatenate([mu, outlier_mean], axis=2)

      sigmas = [sigma for _ in range(n_clusters)]
      if n_clusters>1:
        sigmas += [outlier_sd*np.ones(sigma.shape)]


      mu = pt.reshape(mu, (ns, mu.shape[2], mu.shape[3]))



      for ind_group,group in enumerate(grouped_data):

        _coords = np.argwhere(group[0]).flatten()
        _index = group[1]
        _data = group[2].values
        _serum_biases = serum_biases[...,_index]


        if n_clusters>1:

          components =\
            [pm.CustomDist.dist(mu[:,i,:][...,_coords][:,None,:] + _serum_biases[...,None],
                                sigmas[i][...,None,None],
                                lower[np.ix_(_index,_coords)][None,...],
                                upper[np.ix_(_index,_coords)][None,...],
                                logp=logp, dist=dist,
                                ndim_supp=1) for i in range(n_clusters+1)]
          pm.Mixture(f"mix{ind_group}", w=w[:,None,:],
                     comp_dists=components)


        else:
          pm.CustomDist(f"mix{ind_group}",
                        mu[:,0,:][...,_coords][:,None,:] + _serum_biases[...,None],
                        sigma[...,None,None],
                        lower[np.ix_(_index,_coords)][None,...],
                        upper[np.ix_(_index,_coords)][None,...],
                        logp=logp, dist=dist, ndim_supp=1)

        value = pt.tensor("value", dtype=float, shape=_data.shape)
        logp_fun =\
          pm.compile_pymc([value],
                          pm.logp(getattr(like, f"mix{ind_group}"), value))
        if ind_group==0:
          logps = logp_fun(_data)
        else:
          logps = np.concatenate([logps, logp_fun(_data)], axis=-1)

  return logps
