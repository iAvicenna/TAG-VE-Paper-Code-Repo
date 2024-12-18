#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:06:10 2024

@author: avicenna
"""

# pylint: disable=bad-indentation
import io
import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFont
from matplotlib.cm import get_cmap
from matplotlib import colors as mplcolors

alpha = 1
default_aa_colors={
  "G":[0.97,0.8,0.57,alpha],
  "A":[0.97,0.8,0.57,alpha],
  "S":[0.97,0.8,0.57,alpha],
  "T":[0.97,0.8,0.57,alpha],
  "C":[0.62,0.98,0.59,alpha],
  "V":[0.62,0.98,0.59,alpha],
  "I":[0.62,0.98,0.59,alpha],
  "L":[0.62,0.98,0.59,alpha],
  "P":[0.62,0.98,0.59,alpha],
  "F":[0.62,0.98,0.59,alpha],
  "Y":[0.62,0.98,0.59,alpha],
  "M":[0.62,0.98,0.59,alpha],
  "W":[0.62,0.98,0.59,alpha],
  "N":[0.99,0.57,0.99,alpha],
  "Q":[0.99,0.57,0.99,alpha],
  "H":[0.99,0.57,0.99,alpha],
  "Z":[0.99,0.57,0.99,alpha],
  "D":[0.99,0.57,0.55,alpha],
  "E":[0.99,0.57,0.55,alpha],
  "B":[0.99,0.57,0.55,alpha],
  "K":[0.5,0.5,1,alpha],
  "R":[0.5,0.5,1,alpha],
  "*":[1/3,1/3,1/3,alpha],
  "X":[1/3,1/3,1/3,alpha],
  ".":[0.73,0.73,0.73,1],
  "":[1/3,1/3,1/3,alpha],
  "-":[0.56,0.56,0.56,alpha],

  }

def resize(xscale, yscale, img=None):
  '''
  resize image by width-> xscale*width, height->yscale*height
  '''

  width, height = img.size
  newsize = (int(xscale*width), int(yscale*height))
  return img.resize(newsize)


def crop(x, y, img=None, img_path=None):

  '''
  x=[x0, x1] and y=[y0, y1] give the cropping box
  '''

  assert img is not None or img_path is not None, "you need to supply either img or img_path"
  assert all(r<=1 for r in x) and all(r<=1 for r in y)
  if img is None:
    img = Image.open(img_path)

  w, h = img.size

  return img.crop((x[0]*w, y[0]*w, x[1]*w,  y[1]*h))


def add_text(texts, rel_positions, img=None, img_path=None, font_size=50,
             font_family="Ubuntu-B.ttf", text_args=None, fill=None,
             margin=None):

  '''
  rel_positions: list of [rel_x, rel_y]
  '''

  if fill is None:
    fill=(0, 0, 0)

  assert img is not None or img_path is not None, "you need to supply either img or img_path"
  assert len(texts) == len(rel_positions)

  if text_args is None:
    text_args = {"fill":(0, 0, 0), "stroke_width":0}

  text_args = dict({}, **text_args)

  if img is None:
    img = Image.open(img_path)

  if margin is not None:
    margin = dict({"top":0, "bottom":0, "right":0, "left":0,
                   "color":"white"}, **margin)

    img = add_margin(img, **margin)

  font = ImageFont.truetype(font_family, font_size)

  I1 = ImageDraw.Draw(img)
  x,y = img.size

  for text,rel_position in zip(texts,rel_positions):
    I1.text((x*rel_position[0], y*rel_position[1]), text,
             font=font, fill=fill)
  return img


def add_border(img=None, img_path=None, border_rel_width=0.01, fill="black",
               **expand_args):

  '''
  add border to an img (given as a PIL image or image_path).
  border_rel_width is width relative to image.
  expand_args is extra arguments to ImageOps from PIL
  '''

  assert img is not None or img_path is not None, "you need to supply either img or img_path"

  if img is None:
    img = Image.open(img_path)

  x,y = img.size

  width = int(np.ceil(border_rel_width*(x+y)/2))

  img_with_border = ImageOps.expand(img, border=width, fill=fill,
                                    **expand_args)

  return img_with_border


def fig2img(fig):
  """Convert a Matplotlib figure to a PIL Image and return it"""

  buf = io.BytesIO()
  fig.savefig(buf)
  buf.seek(0)
  img = Image.open(buf)
  return img


def generate_color_categories(categories, return_hex=True):
  """Generate n colors, useful for coloring different categories.
     author: https://www.andersle.no/
  """

  ncol = len(categories)
  if ncol <= 2:
      colors = ['black','tab:red']
  elif ncol == 3:
      colors = ['black','tab:red','tab:blue']
  elif ncol == 4:
      colors = ['black','tab:red','tab:blue','tab:green']
  elif 3 < ncol <= 10:
      colors = list(get_cmap('tab10').colors)
  elif 10 < ncol <= 20:
      colors = list(get_cmap('tab20c').colors)
  elif 20 < ncol <= 33:
       colors1 = list(get_cmap('tab20c').colors)
       colors2 = list(get_cmap('tab10').colors)
       colors3 = list(get_cmap('tab20b').colors)
       colors4 = list(get_cmap('tab20').colors)

       colors = colors1 + [colors2[i] for i in [5,6,8,9]] +\
           [colors3[i] for i in [9,10,11,13,14,15,16,17]] +\
               [colors4[i] for i in [11]]


  elif 33 < ncol <= 256:
      cmap = get_cmap(name='viridis')
      colors = cmap(np.linspace(0, 1, ncol))
  else:
      raise ValueError('Maximum 256 categories but number of categories was '
                       f'{len(categories)}.')
  color_map = {}
  for i, key in enumerate(categories):
      if not return_hex:
          color_map[key] = colors[i]
      else:
          color_map[key] = mplcolors.to_hex(colors[i])

  return color_map


def add_margin(img, top=0, right=0, bottom=0, left=0, color="white"):
  '''
  add margin to image by creating an new image with top, right, bottom, left
  padding added. img can be PIL Image or string
  '''

  if isinstance(img,str):
    pil_img = Image.open(img).convert('RGBA')
  elif isinstance(img,Image.Image):
    pil_img = img
  else:
    pil_img = fig2img(img)

  width, height = pil_img.size
  new_width = width + right + left
  new_height = height + top + bottom
  result = Image.new(pil_img.mode, (new_width, new_height), color)
  result.paste(pil_img, (left, top))

  return result


def combine_images(figures, nrows, ncols, xpad=0, ypad=0, xoffset=0, yoffset=0,
                   background_color=(255,255,255,0), _add_border=False,
                   border=3, border_fill='#AEABAB', equalize=False, xscale=1,
                   yscale=1):

  '''
  combine images given as a list of PIL images or paths into a grid
  with nrows rows and ncols cols. xoffset and yoffset give global offset from
  left margin and top margin where as xpad, ypad give padding between
  images. _add_border can be used to add border to each image.
  if equalize=True, then the resulting image is made square. xscale
  and yscale can be used to rescale the resulting combined image.
  '''

  images = [Image.open(fig).convert('RGBA') if isinstance(fig,str)
            else fig if isinstance(fig,Image.Image)
            else fig2img(fig) for fig in figures]

  if _add_border:
    images = [add_border(img, border=border, fill=border_fill)
              for img in images]

  new_images = images.copy()
  if xpad!=0 or ypad!=0 or xoffset!=0 or yoffset!=0:

    #nimg%ncols not in [0,ncols-1]
    #int(nimg/ncols) not in [0, 1]
    for nimg,img in enumerate(images):

      col = nimg%ncols
      row = int(nimg/ncols)

      if col not in [0,ncols-1]:
        left,right = 0,xpad
      elif col==0:
        left,right = 5 + xoffset, xpad
      elif col==ncols-1:
        left,right = 0, 5

      if row not in [0, nrows-1]:
        top,bottom = 0,ypad
      elif row==0:
        top,bottom = 5 + yoffset, ypad
      elif row==nrows-1:
        top,bottom = 0, 5

      new_images[nimg] = add_margin(img, top, right, bottom, left,"white")


  widths, heights = zip(*(i.size for i in new_images))
  widths = list(widths)
  heights = list(heights)

  if equalize:
    widths = [max(widths) for _ in range(nrows*ncols)]
    heights = [max(heights) for _ in range(nrows*ncols)]

  widths += [0]*(nrows*ncols-len(widths))
  heights += [0]*(nrows*ncols-len(heights))

  x_offsets=np.zeros((nrows,ncols))
  y_offsets=np.zeros((nrows,ncols))

  fig_width = 0
  fig_height = 0

  heights = np.reshape(np.array(heights),(nrows,ncols))
  widths = np.reshape(np.array(widths),(nrows,ncols))

  for row in range(nrows):
      heights[row,:] = np.max(heights[row,:])

  for col in range(ncols):
      widths[:,col] = np.max(widths[:,col])

  fig_height = np.max(np.sum(heights,axis=0))
  fig_width  = np.max(np.sum(widths,axis=1))


  for col in range(1,ncols):
      x_offsets[:,col] += x_offsets[:,col-1] + widths[:,col-1]

  for row in range(1,nrows):
      y_offsets[row,:] += y_offsets[row-1,:] + heights[row-1,:]


  new_im = Image.new('RGB', (fig_width, fig_height),
                     color=background_color)


  x_offsets = x_offsets.flatten().astype(int)
  y_offsets = y_offsets.flatten().astype(int)

  for ind,im in enumerate(new_images):
    new_im.paste(im, (x_offsets[ind], y_offsets[ind]))

  if xscale!=1 or yscale!=1:
    new_im = resize(xscale, yscale, new_im)

  return new_im
