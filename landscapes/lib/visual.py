#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:29:50 2024

@author: avicenna
"""

# pylint: disable=bad-indentation


import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

eps = {
  "grid":0,
  "ag":0.006,
  "contour":0.005
  }


#following are default plotly settings and their explanation is taken from:
#https://plotly.com/python/v3/3d-surface-lighting/
default_surface_properties = {
  "ambient":0.8, #the default amount of light in the room
  "roughness":0.5, #amount of light scattered.
  "diffuse":0.8, #By using Diffuse the light is reflected at many angles rather than just one angle.
  "fresnel":0.2, #wash light over area of plot
  "specular":0.05, #bright spots of lighting in your plot
  }


default_camera = {"up":{'x':0, 'y':0, 'z':1},
                  "center":{'x':0, 'y':0, 'z':0},
                   "eye":{'x':1.25, 'y':1.25, 'z':1.25}}



def plot_landscape(mean_landscape, x,y, base_map, impulses=None,
                   title=None, lld=-1, max_z=None, camera=None, surface_props=None,
                   figsize=None, xbuffer=0, ybuffer=0,
                   antigen_label_shifts=None, put_ticks=True,
                   aspect_ratio=None):

  '''
  mean_landscape gives the height of a landscape at each point on a grid
  where the grid coordinates are given by x and y. So if x.size=y.size=N
  then mean_landscape must be N x N.

  base_map must be a dictionary which contains ag_colours, ag_coordinates,
  ag_names.

  impulses can be used to show the data that landscape tries to fit. it should
  be a list of height values with length equal to number of antigens.

  xbuffer and ybuffer represent how much beyond the map limits to extend the
  figure.

  antigen_label_shifts can be used to shift the location of antigens and it
  should be a dictionary mapping antigen names to a dictionary which has
  members xshift and yshift. If not provided they are both equal to -10
  and labels are plotted to a corner of the antigen centers.
  '''

  if figsize is None:
    figsize=(700, 450)
  else:
    assert isinstance(figsize, tuple) and len(figsize)==2

  if aspect_ratio is None:
    aspect_ratio = [1, 1, 1]
  else:
    assert isinstance(aspect_ratio, list) and len(aspect_ratio)==3

  antigen_colours = base_map["ag_colours"]
  map_coordinates = np.array(base_map["ag_coordinates"])
  ag_names = base_map["ag_names"]

  lims = _get_base_lims(map_coordinates, xbuffer, ybuffer)
  if max_z is None:
    max_z = np.ceil(np.nanmax(mean_landscape))+1

  min_z = lld

  lims = np.concatenate([lims, np.array([min_z,max_z])[:,None]], axis=1)
  fig = _add_landscape(x,y, mean_landscape, surface_props)

  _add_map(map_coordinates, ag_names, antigen_colours, lims, fig,
           antigen_label_shifts)


  if impulses is not None:
     _add_impulses(map_coordinates, impulses, lld, fig)

  fig =_setup_scene(fig, lims, camera, title, figsize, put_ticks,
                    aspect_ratio)

  return fig


def _add_landscape(x, y, z, surface_props):

  if surface_props is None:
    surface_props = {}

  surface_props = dict(default_surface_properties, **surface_props)

  N = x.shape[0]
  cmap = plt.get_cmap("tab10")
  make_int = np.vectorize(int)
  surf_colours = make_int(256*np.array(cmap(1)[0:3])).reshape((1, 1,-1))\
    .repeat(N, axis = 0).repeat(N, axis =1)
  z = np.reshape(z, (N,N)).copy()
  #z[z<lld-0.5] = np.nan
  fig = go.Figure(data=go.Surface(x=x, y=y, z=z, surfacecolor=surf_colours,
                                  opacity=0.5, showscale=False,
                                  lighting=surface_props))

  return fig


def _setup_scene(fig, lims, camera, title, figsize, put_ticks, aspect_ratio):

  if camera is None:
    camera = {}

  if title is None:
    title = {}
  else:
    title={'text': title, 'x':0.8, 'y':0.8}

  camera = dict(default_camera, **camera)

  #_add_base_plane(fig, lims)

  _add_base_grid(fig, lims)

  if put_ticks:
    ticktext = np.round(10*2**np.arange(lims[0,2], lims[1,2]+1, 1),2)
    showticklabels = True
  else:
    ticktext = None
    showticklabels = False

  fig.update_layout(
      title=title,
      autosize=False,
      width=figsize[0],
      height=figsize[1],
      margin={'l':0, 'r':0, 't':0, 'b':0},
      scene={
          "aspectratio":{'x':aspect_ratio[0], 'y':aspect_ratio[1], 'z':aspect_ratio[2]},
          "zaxis":{
              "showbackground" : False,
              "zerolinecolor" : "white",
              "range" : lims[:,2]+np.array([-0.01,0.01]),
              "autorange" : False,
              "tickvals" : np.arange(lims[0,2], lims[1,2]+1, 1),
              "ticktext" : ticktext,
              "title" : "",
              "gridcolor" : "lightgray",
              "tickfont" : {"color":"rgb(10,10,10)","family":"Ubuntu"},
              "tickangle" : 0,
              "showticklabels" : showticklabels
          },
          "xaxis":{
              "range" : lims[:,0],
              "tickvals" : [lims[0,0], lims[1,0]],
              "title" : "",
              "autorange" : False,
              "showticklabels" : False,
              "showgrid" : True,
              "zeroline" : False,
              "showline" : False,
              "gridcolor" : "lightgray",
              "backgroundcolor" : "rgb(255, 255, 255)"
          },
          "yaxis":{
              "range" : lims[:,1],
              "tickvals" : [lims[0,1], lims[1,1]],
              "title" : "",
              "autorange" : False,
              "showticklabels" : False,
              "showgrid" : True,
              "zeroline" : False,
              "showline" : False,
              "gridcolor" : "lightgray",
              "backgroundcolor" : "rgb(255, 255, 255)"
          },
      },
      showlegend=False
  )

  fig.update_layout(scene_camera=camera, title=title)

  return fig


def _add_map(map_coordinates, names, colors, lims, fig,
             antigen_label_shifts):


  marker_dict = {"size":1, "color":"black",
                 "line":{"width":2, "color":"Black"}}

  annotations = []


  for i in range(map_coordinates.shape[0]):

    name = names[i]

    if name not in antigen_label_shifts:
      shift = {"xshift":-10, "yshift":-10}
    else:
      shift = antigen_label_shifts[name]

      if "xshift" not in shift:
        shift["xshift"] = -10
      if "yshift" not in shift:
        shift["yshift"] = -10


    center = map_coordinates[i,:]

    vertices = np.array([center +  0.5*np.array([np.cos(t), np.sin(t)])
                         for t in np.arange(0, 6.5, 0.25)])

    zval = eps["ag"] + lims[0,2]

    fig.add_trace(go.Mesh3d(x=np.round(vertices[:,0],2),
                            y=np.round(vertices[:,1],2),
                            z=zval*np.ones((vertices.shape[0],))+0.012,
                            color=colors[i]))

    fig.add_trace(go.Scatter3d(x=np.round(vertices[:,0],2),
                               y=np.round(vertices[:,1],2),
                               z=zval*np.ones((vertices.shape[0],))+0.012,
                               marker=marker_dict, mode="lines"))


    annotations.append(
          {
          "showarrow":False,
          "x":center[0],
          "y":center[1],
          "z":lims[0,2],
          "text":name,
          "xanchor":"left",
          "borderwidth":2,
          "font":{"color":"black", "size":12, "family":"Ubuntu"},
          "opacity":1,
          **shift})

  fig.update_layout(scene={"annotations":annotations})


def _add_impulses(map_coordinates, impulses, lld, fig):

  marker_dict = {"size":10, "color":"black"}

  fig.add_trace(go.Scatter3d(x=map_coordinates[:,0], y=map_coordinates[:,1],
                             z=impulses, marker=marker_dict, mode="markers"))

  nantigens = map_coordinates.shape[0]
  line = {"color":"rgb(10,10,10)"}

  for i in range(nantigens):

    fig.add_trace(go.Scatter3d(x=map_coordinates[[i,i],0], y=map_coordinates[[i,i],1],
                               z=[lld, impulses[i]], line=line,
                               marker=marker_dict, mode="lines"))


def _add_base_plane(fig, lims):

  lld = lims[0,2]
  x, y = np.meshgrid(np.linspace(*lims[:,0], 100), np.linspace(*lims[:,1], 100))

  #z plane
  fig.add_traces(
      go.Surface(
          x=x,
          y=y,
          z=lld*np.ones(x.shape),
          colorscale=[
              [0, "rgba(234, 234, 254, 1)"],
              [1, "rgba(234, 234, 254, 1)"]
          ],
          opacity=1,
          showscale=False,
      )
  )


def _add_base_grid(fig, lims):

  lld = lims[0,2]

  line_marker = {"color":"lightgray", "width":2}

  X, Y = np.meshgrid(np.arange(lims[0,0],lims[1,0]+1, 1),
                     np.arange(lims[0,1], lims[1,1]+1, 1))
  zval = lld + eps["grid"]

  for xx, yy in zip(X, Y):
    fig.add_scatter3d(x=xx, y=yy, z=zval*np.ones(xx.shape), mode='lines',
                      line=line_marker, opacity=1)

  #Define the second family of coordinate lines
  Y, X = np.meshgrid(Y, X)
  for xx, yy in zip(X, Y):
    fig.add_scatter3d(x=xx, y=yy, z=zval*np.ones(xx.shape), mode='lines',
                      line=line_marker, opacity=1)

  line_marker["color"] = "darkgray"

  fig.add_scatter3d(x=X[:,0], y=np.ones(xx.shape)*lims[0,1]+0.05,
                    z=zval*np.ones(xx.shape)+0.01, mode='lines',
                    line=line_marker, opacity=1)
  fig.add_scatter3d(x=X[:,0], y=np.ones(xx.shape)*lims[1,1]-0.05,
                    z=zval*np.ones(xx.shape)+0.01, mode='lines',
                    line=line_marker, opacity=1)
  fig.add_scatter3d(y=Y[0,:], x=np.ones(xx.shape)*lims[0,0]+0.05,
                    z=zval*np.ones(xx.shape)+0.01, mode='lines',
                    line=line_marker, opacity=1)
  fig.add_scatter3d(y=Y[0,:], x=np.ones(xx.shape)*lims[1,0]-0.05,
                    z=zval*np.ones(xx.shape)+0.01, mode='lines',
                    line=line_marker, opacity=1)


def _get_base_lims(coordinates, xbuffer=0, ybuffer=0):
  """
  column 1 is xlims
  column 2 is ylims
  """

  centered_coordinates = coordinates - np.mean(coordinates,axis=0)

  max_val = np.ceil(np.max(np.abs(centered_coordinates))) + 0.5

  lims=\
    np.array([np.mean(coordinates,axis=0) - max_val - xbuffer,
              np.mean(coordinates,axis=0) + max_val + ybuffer])

  return lims
