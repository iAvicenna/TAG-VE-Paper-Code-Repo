#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:49:59 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, import-error, wrong-import-position

# pylint: disable=unsubscriptable-object
# pylint for some reason thinks pymc random variables are unsubscriptable
# which is wrong

import pymc as pm
import numpy as np
import pytensor.tensor as pt
import pandas as pd

default_prior_params = {"slope_sd":0.25, "slope_mu":0.8, "height_sd":0.25,
                        "apex_sd":0.25, "noise_mu":0.75, "noise_sd":0.25,
                        "sr_bias_sigma":0.25, "nu_mu":10, "nu_sigma":3}



def centre_data(flat, level_sets, I_sr_levels):
  '''
  centres the data by subtracting from each row its mean difference
  from the nonparametric gmt of the data.
  '''

  wide = flat.copy()

  wide = pd.pivot(wide, columns="ag_name", index="sr_name",
                  values="titre")

  wide = wide.loc[level_sets["sr"], level_sets["ag"]]

  gmt_pt_est = np.nanmean(wide, axis=0)

  sr_bias_ptest = np.nanmean(wide - gmt_pt_est[None,:], axis=1)

  flat.loc[:,"titre"] -= sr_bias_ptest[I_sr_levels].astype(float)
  flat.loc[:,"lower"] -= sr_bias_ptest[I_sr_levels].astype(float)
  flat.loc[:,"upper"] -= sr_bias_ptest[I_sr_levels].astype(float)

  return flat




def cone_mixture_model(ncones, flat, ag_to_coordinates, apex_mixture_mus=None,
                       apex_mixture_weights=None, height_mixture_mus=None,
                       height_mixture_weights=None, initvals=None,
                       combine_fun="gs", ag_order=None, prior_params=None):

  '''
  This function returns a model that fits cones to a given titre data
  (stored in the input flat) using antigen coordinate information given
  by ag_to_coordinates. number of cones to fit is given by ncones.
  It is assumed that titre, lower and upper values in flat is in log2
  units. The data is centred by subtracting from each row its mean
  difference with respect to the column mean of the all data.

  flat is a table of titres where each row should contain the information
  of ag_name, sr_name, lower and upper values for titre.
  For a titre of the form <val, val==lower, for a titre of the form
  >val, val==upper.

  In addition the table can also contain sr_group info. If it does then
  each set of data belonging to different sr_groups is fitted independent
  cones. Only interaction between different sr_groups is measurement noise
  sd which is fitted through the whole data. Number of sr_groups will be
  denoted by nsrgroups. This however assumes that each sr_group is fitted
  with the same number of cones.

  The likelihood for titres is a censored student-T

  The flat table might contain multiple sera titrated against the same
  antigens. In this case serum reactivity bias (after mean centring the data)
  is also determined for each individual serum and the titres of the given
  sera can be thought of as:

    serum titre value at antigen A = height of landscape at antigen A
    + serum reactivity value

  sr_biases are drawn from a normal with 0.

  To fit a cone, its base point apex_coordinates, slope and height are fitted.
  apex_coordinates and heights priors are constructed as weighted mixture of
  normals. In the case of apex_coordinates, if not specified, apex_coordinate
  normals' mus coincide with the coordinates of antigens in the table with the
  sd given by apex_sd. If specified apex_mixture_mus should be of the form
  nsrgroups x nnormals x 2 for nnormals>0 (where nnormals represents number
  of normals included in the mixture). The apex_mixture_weights should be of
  the form nsrgroups x ncones x nnormals. So in summary each cone uses the same
  apex_mixture_mus as prior for its apex but has its own weight. The weight
  allows you to focus on different antigen regions for different cones. apex_sd
  and height_sd determine the standard deviation of the normals involved in
  the mixture.

  Example:
    ncones = 2
    nsrgroups = 1
    nnormals = 3
    apex_mixture_mus = np.array([[[1, 0], [2, 0], [3, 0]]]) (has shape 1,3,2)
    apex_mixture_weights = np.array([[[10, 5, 1], [0, 0, 1]]]) (has shape 1,2,3)
    apex_sd = 0.1

  In this case the apex_prior for the first cone is a mixture of three 2D
  (independent in each corodinate) normals: first one centered at [1,0] with
  weight 10, second one centred at [2,0] with weight 5 and third one
  at [3,0] with weight 1 (with a sd=0.1 around each) The second cone has a
  weight of 0 for the first two coordinates so its apex is a single normal
  centered at [3,0] with sd=0.1 (because other weights are 0). If there are
  multiple srgroups then different mus and weights need to specified
  for each.

  Similar rules apply to height_mixtures. If not specified it is determined
  based on the values in the table flat. If specified it should be of the
  form nsrgroups x nnormals  (nnormals can be different from the one for mu)
  and similarly height_mixture_weights should be of the form
  nsrgroups x ncones x nnormals.

  combine_fun is used to determine how the multiple cones fitted are combined
  together to form the final landscape, it is one of the following:
  max, gmt, amt, sum.

  Following posteriors are reported in the returned idata (among some others
  which wont be of much use to the end user):

    - sr_biases (serum reactivity biases), sr_bias_sigma, titre_sd
      (if number of sera >1)
    - apex_coordinates, heights, slopes
    - radii are the radii of each of the sampled cones projection to the plane
      at z = lld (lower limit of detection). lld is an input with default value 0
      (corresponding to a titre of 10)
    - combined_landscape (height values of the landscape at the given antigens)
    - individual_cones (height values of the individual cones at the given
                        antigens)
    - observed_gmts_wosrbias: these are the mean of observed_titres - sr_bias,
    where mean is taken across the sera. this can be thought of as the gmt
    of the titres at each antigen after removing serum reactivity biases.
    The better your fit is the more close these values are to your combined_landscape
    values.
    - fitted_titres are the estimated titres for each of your serum at each of
    the antigens. the better your fit is, closer these values are to the observed
    titres.

  Technical information:

    In order to break the invariance of the likelihood with respect to permutation
    of the cones, an ordering constraint is placed on the xcoordinates of the apexes.
    This means that if you are sampling xcoordinates for cone 1, then it should
    always be increasing such as 0.1, 0.5, 0.8. In a situation where you are fitting
    for instance two cones with true apex coordinates z0 and z1, it is equivalent
    from likelihood perspective for the first cone to sample z0 and second
    one to sample z1 and the other way around. Ordering remedies this up to a
    point, though it is not a silver bullet. Trying to do statistics on
    apex coordinates is therefore liable to faulty inference. Luckily this is
    not the point here and we are only interested in the resulting combined cone
    which is agnostic to sampling order of apexes. If you really want to visualize
    individual cones instead of the combined, one way to do this is to use
    the combine method combine_fun = max along with sample_grid_landscape
    below. This will not give you individual cone apexes however by the
    shape of the sampled max landscape, you will have a pretty good
    idea about where they should be (you could compute the local maximas on
    a fine enough grid and sample apexes if you really want to).
  '''

  if prior_params is None:
    prior_params = {}

  prior_params = dict(default_prior_params, **prior_params)

  assert ncones>0
  flat = flat.copy()[~flat.titre.isin([np.nan])] #drop unmeasured

  level_sets, flat = _mixture_landscape_levels(flat, ag_order)

  nsera = len(level_sets["sr"])
  nsrgroups = len(level_sets["sr_group"])
  I_ag_levels = np.array([level_sets["ag"].index(x) for x in flat["ag_name"]])
  I_sr_levels =  np.array([level_sets["sr"].index(x) for x in flat["sr_name"]])
  I_sr_group_levels =  np.array([level_sets["sr_group"].index(x)
                                     for x in flat["sr_group"]])
  I = (I_ag_levels, I_sr_group_levels)

  flat = centre_data(flat, level_sets, I_sr_levels)


  nantigens = len(level_sets["ag"])
  measurement_coordinates = np.array([ag_to_coordinates[ag] for ag in level_sets["ag"]])


  apex_mixture_mus, apex_mixture_weights=\
    _mixture_landscape_apex_mus_and_weights(apex_mixture_mus,
                                            apex_mixture_weights,
                                            measurement_coordinates,
                                            nsrgroups, ncones,nantigens)

  height_mixture_mus, height_mixture_weights=\
    _mixture_landscape_height_mus_and_weights(flat, height_mixture_mus,
                                              height_mixture_weights,
                                              nsrgroups, ncones, level_sets["ag"])

  initvals=\
    _mixture_landscape_initial_conditions(initvals, apex_mixture_mus,
                                          nsrgroups, ncones)

  obs_coords = [f"{x}_{y}" for x,y in zip(flat.loc[:,"sr_name"],
                                          flat.loc[:,"ag_name"])]


  coords = {
    "serum":level_sets["sr"],
    "cone":list(range(ncones)),
    "dim":["x","y"],
    "sr_group":level_sets["sr_group"],
    "ag":level_sets["ag"],
    "obs_name":obs_coords
    }

  meta={}
  meta["initvals"] = initvals
  meta["level_sets"] = level_sets
  meta["flat"] = flat


  with pm.Model(coords=coords) as model:

    #DATA
    pm.Data("I_sr_levels", I_sr_levels)
    pm.Data("I_ag_levels", I_ag_levels)
    pm.Data("I_sr_group_levels", I_sr_group_levels)
    pm.Data("lower", flat.loc[:,"lower"].values)
    pm.Data("upper", flat.loc[:,"upper"].values)
    #sr_bias_pt_est = pm.Data("sr_bias_pt_est", sr_bias_pt_est)

    #PRIORS

    nu = pm.InverseGamma("nu", mu=prior_params["nu_mu"],
                         sigma=prior_params["nu_sigma"])


    if nsera>1:

      sr_biases = pm.Normal("sr_biases", 0,
                            prior_params["sr_bias_sigma"],
                            shape=nsera, dims="serum")

      titre_sd = pm.InverseGamma("titre_sd", prior_params["noise_mu"],
                                 prior_params["noise_sd"],
                                 dims="ag",
                                 shape=nantigens)


    else:
      titre_sd = pm.math.constant(1, "titre_sd")
      sr_biases = pm.math.constant(np.array([0]), "sr_biases")

    #preparation for definining the apex_coordinates and height mixture of
    #normals prior
    for i in range(nsrgroups):
      # these create ncones copies of xcomps and ycomps to be used
      # for each cone but based on same apex_mixture_mus for each cone

      _xcoordinates=\
      pm.NormalMixture(f"xcoordinates{i}", w=apex_mixture_weights[i,...],
                       mu=apex_mixture_mus[i,:,0],
                       sigma=prior_params["apex_sd"],
                       transform=pm.distributions.transforms.ordered,
                       dims=["cone"], size=(ncones))

      _ycoordinates=\
      pm.NormalMixture(f"ycoordinates{i}", w=apex_mixture_weights[i,...],
                       mu=apex_mixture_mus[i,:,1],
                       sigma=prior_params["apex_sd"],
                       size=(ncones), dims=["cone"])



      _heights=\
      pm.NormalMixture(f"heights{i}", w=height_mixture_weights[i,:,:],
                       mu=height_mixture_mus[i,:],
                       sigma=prior_params["height_sd"],
                       dims=["cone"], size=(ncones))

      if i==0:
        xcoordinates = _xcoordinates[:,None]
        ycoordinates = _ycoordinates[:,None]
        heights = _heights[:,None]
      else:
        xcoordinates = pt.concatenate([xcoordinates, _xcoordinates[:,None]], axis=1)
        ycoordinates = pt.concatenate([ycoordinates, _ycoordinates[:,None]], axis=1)
        heights = pt.concatenate([heights, _heights[:,None]], axis=1)

    apex_coordinates =\
      pm.Deterministic("apex_coordinates",
                       pt.stack([xcoordinates, ycoordinates], axis=1),
                       dims=["cone","dim","sr_group"])
    pm.Deterministic("heights", heights,
                     dims=["cone","sr_group"])

    slopes = pm.InverseGamma("slopes", prior_params["slope_mu"],
                             prior_params["slope_sd"],
                             shape=(ncones, nsrgroups),
                             dims=["cone","sr_group"])

    #TRANSFORMED PRIORS

    #difference between apex_coordinates and all the measurement_coordinates
    dif_coords = apex_coordinates[:,None,:,:] - measurement_coordinates[None, :, :, None]

    #distance between apex corodinates and all the measurement_coordinates
    #norm_dif_coords has shape ncones, nag, nsrgroups
    norm_dif_coords = pm.math.sqrt(pm.math.sqr(dif_coords[:,:,0,:])
                                   + pm.math.sqr(dif_coords[:,:,1,:]))

    individual_cones = heights[:,None,:] - slopes[:,None,:]*norm_dif_coords

    combined_landscape = _combine(individual_cones,  combine_fun)


    fitted_titres = pm.Deterministic("fitted_titres",
                                     combined_landscape[I] +
                                     sr_biases[I_sr_levels])

    #LIKELIHOOD
    st_dist = pm.StudentT.dist(mu=fitted_titres, nu=nu,
                               sigma=titre_sd[I_ag_levels])


    pm.Censored("obs", st_dist, lower=flat.loc[:,"lower"].values,
                upper=flat.loc[:,"upper"].values,
                observed=flat.loc[:,"titre"].values.astype(float),
                dims="obs_name")



  return model, meta



def sample_observables(model, idata,  meta, combine_fun="gs",
                       sample_args=None):

  '''
  After the model is sampled, you can use this function to do posterior
  sampling of mean observed titres without bias. use the same combine function
  used as in the model
  '''


  if sample_args is None:
    sample_args = {}

  sample_args = dict({"extend_inferencedata":True,
                      "predictions":True})

  level_sets = meta["level_sets"]
  base_map = meta["base_map"]

  ag_to_coordinates = dict(zip(base_map["ag_names"],
                               base_map["ag_coordinates"]))
  measurement_coordinates = np.array([ag_to_coordinates[ag] for ag in level_sets["ag"]])


  var_names = ["ag_mean_titres"]

  with model:

    apex_coordinates = model.apex_coordinates
    heights = model.heights
    slopes = model.slopes

    #difference between apex_coordinates and all the measurement_coordinates
    dif_coords = apex_coordinates[:,None,:,:] - measurement_coordinates[None, :, :, None]

    #distance between apex corodinates and all the measurement_coordinates
    #norm_dif_coords has shape ncones, nag, nsrgroups
    norm_dif_coords = pm.math.sqrt(pm.math.sqr(dif_coords[:,:,0,:])
                                   + pm.math.sqr(dif_coords[:,:,1,:]))

    individual_cones = heights[:,None,:] - slopes[:,None,:]*norm_dif_coords

    pm.Deterministic("ag_mean_titres",
                     _combine(individual_cones,  combine_fun),
                     dims=["ag","sr_group"])


    pm.sample_posterior_predictive(idata, var_names=var_names, **sample_args)





def sample_grid_landscape(model, idata, flat, gw, combine_fun,
                          ag_coordinates, sample_args=None,
                          buffer=3):

  '''
  After model sampling is done, this function allows you to do posterior
  sampling to construct landscapes. If you use the same combine_fun as
  the model, you can get your HDI for the landscape heights at each position.
  If you use combine_fun = "max", it will give you an idea about where
  each cone in the mixture is situated.

  gw = width of the grid to sample
  buffer = how much beyond the map to sample.
  '''

  if sample_args is None:
    sample_args = {}

  sample_args = dict({"extend_inferencedata":True,
                      "predictions":True})

  base_lims = _get_base_lims(ag_coordinates, buffer)
  grid_xcoords = np.arange(base_lims[0,0], base_lims[1,0]+gw, gw)
  grid_ycoords = np.arange(base_lims[0,1], base_lims[1,1]+gw, gw)
  grid = np.meshgrid(grid_xcoords, grid_ycoords)


  grid = list(zip(grid[0].flatten(), grid[1].flatten()))
  model.coords["grid"] = range(len(grid))

  grid = np.array(grid)
  var_names = ["combined_grid_landscape_values",
               "observed_gmts_wosrbias"]


  with model:

    apex_coordinates = model.apex_coordinates
    heights = model.heights
    slopes = model.slopes
    I_sr_levels = np.array(idata["constant_data"]["I_sr_levels"])
    I_ag_levels = np.array(idata["constant_data"]["I_ag_levels"])
    I_sr_group_levels = np.array(idata["constant_data"]["I_sr_group_levels"])

    if "sr_biases" in dir(model):
      sr_biases = model.sr_biases
    else:
      sr_biases = np.array([0])

    _sample_grid_landscape(grid, apex_coordinates, heights, slopes,
                           combine_fun)

    _sample_landscape_gmts(flat, sr_biases, I_sr_levels,
                            I_ag_levels, I_sr_group_levels)

    pm.sample_posterior_predictive(idata, var_names=var_names, **sample_args)

  return grid


def _sample_grid_landscape(grid, apex_coordinates, heights, slopes,
                           combine_fun):


  dif_coords = apex_coordinates[:,None,:,:] - grid[None, :, :, None]

  norm_dif_coords = pm.math.sqrt(pm.math.sqr(dif_coords[:,:,0,:])
                                 + pm.math.sqr(dif_coords[:,:,1,:]))

  individual_cones = heights[:,None,:] - slopes[:,None,:]*norm_dif_coords

  pm.Deterministic("combined_grid_landscape_values",
                   _combine(individual_cones, combine_fun),
                   dims=["grid","sr_group"])


def _sample_landscape_gmts(flat, sr_biases, I_sr_levels, I_ag_levels,
                           I_sr_group_levels=None):

  sampled_titres_wobias = flat.loc[:,"titre"].values.astype(float) -\
    sr_biases[I_sr_levels]

  nantigens = len(set(I_ag_levels))

  if I_sr_group_levels is None:
    I_sr_group_levels = np.array([0 for _ in range(flat.shape[0])])

  n_sr_groups = len(set(I_sr_group_levels))

  I = [np.argwhere((I_ag_levels==i)&(I_sr_group_levels==j)).flatten()
       for j in range(n_sr_groups) for i in range(nantigens) ]

  pm.Deterministic("observed_gmts_wosrbias",
      pt.reshape(pt.stack([sampled_titres_wobias[i].mean() if len(i)>0
                             else np.nan for i in I]),
                  (n_sr_groups, nantigens)),
      dims=["sr_group","ag"])



def _combine(tensor, combine_fun):

  if combine_fun=="max":
    return tensor.max(axis=0)
  if combine_fun=="am":
    return pm.math.log(pm.math.exp(tensor).mean(axis=0))
  if combine_fun=="gm":
    return tensor.mean(axis=0)
  if combine_fun=="as":
    return pm.math.log(pm.math.exp(tensor).sum(axis=0))
  if combine_fun=="gs":
    return tensor.sum(axis=0)

  raise ValueError("combine_fun can only be max, am, gm, as, gs but was "
                   f"{combine_fun}")


def _mixture_landscape_apex_mus_and_weights(apex_mixture_mus,
                                            apex_mixture_weights,
                                            measurement_coordinates,
                                            nsrgroups, ncones, nantigens):


  if apex_mixture_mus is None:
    apex_mixture_mus = np.tile(measurement_coordinates, (nsrgroups, 1, 1))
  else:
    assert (apex_mixture_mus.shape[0],apex_mixture_mus.shape[2])\
      == (nsrgroups, 2)

  if apex_mixture_weights is None:
    apex_mixture_weights = np.ones((ncones, apex_mixture_mus.shape[0]))/nantigens
    apex_mixture_weights = np.tile(apex_mixture_weights, (nsrgroups,1,1))
  else:
    assert apex_mixture_weights.shape == (nsrgroups, ncones, apex_mixture_mus.shape[1])

  apex_mixture_weights =\
    apex_mixture_weights/np.sum(apex_mixture_weights,axis=2)[:,:,None]

  return apex_mixture_mus, apex_mixture_weights


def _mixture_landscape_height_mus_and_weights(flat, height_mixture_mus,
                                              height_mixture_weights,
                                              nsrgroups, ncones, ag_levels):


  nantigens = len(ag_levels)

  observed_vals = flat.loc[:,"titre"].values


  if height_mixture_mus is None:
    ag_to_mean = {ag: observed_vals[flat.ag_name.isin([ag])].mean()
                  for ag in ag_levels}

    height_mixture_mus = np.tile(list(ag_to_mean.values()), (nsrgroups,1))

  else:
    assert height_mixture_mus.shape[0] == nsrgroups


  if height_mixture_weights is None:
    height_mixture_weights = np.ones((ncones,height_mixture_mus.shape[0]))/nantigens
    height_mixture_weights = np.tile(height_mixture_weights, (nsrgroups,1,1))

  else:
    assert height_mixture_weights.shape == (nsrgroups, ncones, height_mixture_mus.shape[1])

  height_mixture_weights =\
    height_mixture_weights/np.sum(height_mixture_weights,axis=2)[:,:,None]


  return height_mixture_mus, height_mixture_weights


def _mixture_landscape_initial_conditions(initvals, apex_mixture_mus, nsrgroups,
                                          ncones):

  #update this where values are set based on apex_mixture_weight sorted highest

  if initvals is None:
    initvals={}

  for i in range(nsrgroups):
    if f"xcoordinates{i}" not in initvals:
      initvals[f"xcoordinates{i}"] =\
       np.sort(apex_mixture_mus[i,:,0])[:ncones]

      if f"ycoordinates{i}" not in initvals:
        initvals[f"ycoordinates{i}"] =\
          apex_mixture_mus[i,np.argsort(apex_mixture_mus[i,:,0]).\
                                   flatten(),1][:ncones]
      else:
        assert initvals[f"ycoordinates{i}"].shape == (ncones,)

    else:
      assert initvals[f"xcoordinates{i}"].shape == (ncones,)
      assert all(np.all(np.argsort(initvals[f"xcoordinates{i}"]) ==
                        np.arange(0, ncones))
                 for i in range(nsrgroups))

  return initvals


def _mixture_landscape_levels(flat, ag_order=None):

  if "sr_group" not in flat:
    flat.loc[:,"sr_group"] =  np.zeros(flat.shape[0])

  if ag_order is None:
    ag_levels = sorted(list(set(flat["ag_name"])))
  else:
    assert all(x in set(flat["ag_name"]) for x in ag_order)
    ag_levels = ag_order

  sr_levels = sorted(list(set(flat["sr_name"])))
  sr_group_levels = sorted(list(set(flat["sr_group"])))

  return {"ag":ag_levels, "sr":sr_levels, "sr_group":sr_group_levels}, flat


def _get_base_lims(coordinates, buffer=0):
  """
  column 1 is xlims
  column 2 is ylims
  """
  centered_coordinates = coordinates - np.mean(coordinates,axis=0)

  max_val = np.ceil(np.max(np.abs(centered_coordinates))) + 0.5

  lims=\
    np.array([np.mean(coordinates,axis=0) - max_val - buffer,
              np.mean(coordinates,axis=0) + max_val + buffer])

  return lims
