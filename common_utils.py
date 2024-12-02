#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 09:48:39 2024

@author: avicenna
"""
# pylint: disable=bad-indentation, line-too-long



import numbers
import numpy as np

_HDI = 0.94
threshold = 0.10
antigens = ['Alpha', 'Beta', 'Delta', 'BA.1', 'BA.5',  'XBB.1.5']
min_non_thresholded = 2


ag_colours = dict(zip(antigens, ['#637939', '#e7ba52', '#d18652',
                                 '#EF3737', '#F08DA5', '#3d656e']))
ag_colours["WT"] = "#211c59"
ag_colours["XBB.1"] = ag_colours["XBB.1.5"]



_code_to_common_name = {
    "AFRIMS_TH_SEARO": "AFRIMS",
    "EMS_NL_EURO": "Erasmus",
    "VIDRL_AU_WPRO": "VIDRL",
    "RLID_AUE_EMRO":"RLID",
    "FIOCRUZ_BS_PAHO":"FIOCRUZ",
    "USCDC_PAHO":"US CDC",
    "NMIMR_GH_AFRO":"NMIMR",
    "AHRI_ZA_AFRO":"AHRI",
    "CUH_DE_EURO":"Charité",
    "UHG_CH_EURO":"Genève",
    "EVC_US_PAHO":"Emory",
    "EVC2_US_PAHO":"Emory\n(in-house)",
    "FIVI_CH_EURO":"Bern",
    "MUI_AT_EURO":"Innsbruck",
    "MUI2_AT_EURO":"Innsbruck\n(in-house)",
    "NICD_SA_AFRO":"NICD",
    "RKI_DE_EURO":"RKI",
    "SPH_HK_WPRO":"HKU",
    }


_code_to_LLD = {
  "AFRIMS_TH_SEARO_mn": 10,
  "EMS_NL_EURO_frnt": 20,
  "VIDRL_AU_WPRO_mn": 10,
  "RLID_AUE_EMRO_prnt":10,
  "FIOCRUZ_BS_PAHO_prnt":[10, 160],
  "USCDC_PAHO_prnt":10,
  "NMIMR_GH_AFRO_mn":40,
  "NMIMR_GH_AFRO_prnt":40,
  "AHRI_ZA_AFRO_frnt":10,
  "CUH_DE_EURO_prnt":40,
  "UHG_CH_EURO_frnt":10,
  "EVC_US_PAHO_frnt":10,
  "EVC2_US_PAHO_frnt":10,
  "FIVI_CH_EURO_mn":[28.3, 56.6],
  "MUI_AT_EURO_frnt":10,
  "MUI2_AT_EURO_frnt":10,
  "NICD_SA_AFRO_pnt":20,
  "RKI_DE_EURO_prnt":10,
  "RKI_DE_EURO_pnt":40,
  "SPH_HK_WPRO_mn":10,
  "SPH_HK_WPRO_prnt":10,
  }


_code_to_ULD = {
  "AFRIMS_TH_SEARO_mn": 163840,
  "EMS_NL_EURO_frnt": np.inf,
  "VIDRL_AU_WPRO_mn": 2560,
  "RLID_AUE_EMRO_prnt":320,
  "FIOCRUZ_BS_PAHO_prnt":5120,
  "USCDC_PAHO_prnt":7290,
  "NMIMR_GH_AFRO_mn":5120,
  "NMIMR_GH_AFRO_prnt":5120,
  "AHRI_ZA_AFRO_frnt":np.inf,
  #dilution range runs upto 3200 but they allow parameter estimates without
  #boundary
  "CUH_DE_EURO_prnt":10240,
  "UHG_CH_EURO_frnt":81920,
  "EVC_US_PAHO_frnt":43740,
  "EVC2_US_PAHO_frnt":43740,
  "FIVI_CH_EURO_mn":2560,
  "MUI_AT_EURO_frnt":163840,
  "MUI2_AT_EURO_frnt":163840,
  "NICD_SA_AFRO_pnt":81920,
  "RKI_DE_EURO_prnt":20480,
  "RKI_DE_EURO_pnt":10800,
  "SPH_HK_WPRO_mn":320,
  "SPH_HK_WPRO_prnt":10240
  }


_code_to_info = {
  "EMS_NL_EURO": {"FRNT":{"virus-type":"live", "growth-cell":"Calu-3", "reps":1,
                          "assay-cell":"Vero-TMPRSS2","LLD":20, "ULD":np.nan}},
  "VIDRL_AU_WPRO":{"MN":{"virus-type":"live","growth-cell":"Vero E6-TMPRSS2 / Calu3",
                         "assay-cell":"Vero E6-TMPRSS2", "LLD":10, "ULD":np.nan,
                         "titre-method":"calculated using the Reed and Muench method",
                         "reps":4}},
  "RLID_AUE_EMRO":{"PRNT":{"virus-type":"live", "LLD":10, "ULD":320, "growth-cell":"Vero E6", "reps":2,
  												 "assay-cell":"Vero E6","titre-method":"BioTek Gen5 was used for curve fitting and titre computation"}},
  "FIOCRUZ_BS_PAHO":{"PRNT":{"virus-type":"live","LLD":10,"ULD":5120,"assay-cell":"NA",
                             "reps":"1"}},
  "USCDC_PAHO": {"PRNT":{"virus-type":"live", "LLD":10, "ULD":7290, "reps":1,
                         "assay-cell":"VeroE6/TMPRSS2/ACE2",
                         "notes":"due to problems in generating enough stocks, WA-1 was used instead of Alpha and ND10 titres were measured", "titre-method":"graph-pad was used for calculating titres."}},
  "NMIMR_GH_AFRO": {"MN":{"virus-type":"live","assay-cell":"Vero TMPRSS-2", "reps":1, "LLD":40, "ULD":np.nan},
                    "PRNT":{"virus-type":"live","assay-cell":"Vero TMPRSS-2", "reps":1, "LLD":40, "ULD":np.nan}},
  "AHRI_ZA_AFRO":{"FRNT":{"virus-type":"live", "growth-cell":"VeroE6-TMPRSS2", "assay-cell":"VeroE6-TMPRSS2", "reps":1, "LLD":np.nan, "ULD":np.nan}},
  "AFRIMS_TH_SEARO":{"MN":{"assay-cell":"Vero E6", "growth-cell":"Vero E6",
                           "reps":"8","virus-type":"live", "LLD":10, "ULD":163840,
                           "titre-method":"titre computed using log probit analysis"}},
  "CUH_DE_EURO":{"PRNT":{"growth-cell":"Vero E6", "assay-cell":"Vero E6", "reps":"2", "LLD":40, "ULD":1280,
                         "virus-type":"live", "titre-method":"titres computed via neutcurve package"}},
  "UHG_CH_EURO":{"FRNT":{"assay-cell":"Vero E6/TMPRSS", "reps":"2", "virus-type":"live", "LLD":10, "ULD":np.nan,
                         "titre-method":"titres computed via 4 parameter curve fitting"}},
  "EVC_US_PAHO":{"FRNT":{"virus-type":"live", "assay-cell":"VeroE6 TMPRSS2", "reps":1, "LLD":"10","ULD":np.nan}},
  "FIVI_CH_EURO":{"MN":{"assay-cell":"Vero E6/TMPRSS2", "growth-cell":"Vero E6/TMPRSS2", "reps":"3", "virus-type":"live", "LLD":"28.3 or 56.6", "ULD":np.nan,
                        "titre-method":"titres computed using Spearman and Kärber method"}},
  "MUI_AT_EURO":{"FRNT":{"assay-cell":"Vero-TMPRSS2/ACE2", "reps":"2", "virus-type":"live","LLD":"10","ULD":np.nan,
                         "titre-method":"titres computed using graph-pad non-linear regr."}},
  "NICD_SA_AFRO":{"PNT":{"geowth-cell": "293T/ACE", "assay-cell":"293T/ACE2", "reps":"1", "virus-type":"pseudo", "LLD":"20", "ULD":np.nan,
                         "notes": "titres were 50% reduction of relative light"}},
  "RKI_DE_EURO":{"PRNT":{"assay-cell":"Vero E6", "reps":"2", "virus-type":"live", "LLD":10, "ULD":np.nan,
                         "notes":"Beta and Delta did not produce any plaques", "titre-method":"titre was lowest serum dilution that yields less than %50 of initial"},
                 "PNT":{"assay-cell":"HT1080-ACE2 cells", "reps":"1", "virus-type":"pseudo", "LLD":40, "ULD":np.nan,
                                "notes":"Delta Pseudo lacks G142, G1167 and G1219", "titre-method": "titres computed using graph-pad non-linear regr."}},
  "SPH_HK_WPRO":{"MN":{"assay-cell":"Vero E6/TMPRSS2", "reps":4, "virus-type":"live", "LLD":10, "ULD":320,
                       "titre-method":"titre was the highest serum dilution that completely protected the cells from CPE in half of the wells"},
                 "PRNT":{"assay-cell":"Vero E6/TMPRSS2", "reps":2, "virus-type":"live","LLD":10, "ULD":np.nan,
                         "titre-method":"titre was lowest serum dilution that yields less than %50 of initial"}}}

_code_to_info["EVC2_US_PAHO"] = _code_to_info["EVC_US_PAHO"]
_code_to_info["MUI2_AT_EURO"] = _code_to_info["MUI_AT_EURO"]


def log(x):
  '''
  a log function that can either work with numbers or strings. returns
  log(x/10). if string and thresholded of the form <10 returns log(10/10)
  np.nan or * is returned as np.nan
  '''

  if isinstance(x, numbers.Number):

    if not np.isnan(x):
      return np.log2(x/10)

    return np.nan

  if isinstance(x, str):

    if x[0] in ['<', '>']:
      return np.log2(float(x[1:])/10)

    if x == '*':
      return np.nan

    return np.log2(float(x)/10)

  raise ValueError("x should be numeric or string")


def add_censoring(flat_table, censoring_type):
  '''
  given a table with titre column, adds lower or upper columns to the table.
  these depend on the functions lower or upper below.
  '''

  if censoring_type == "lower":
    fun = lower
  elif censoring_type == "upper":
    fun = upper
  else:
    raise ValueError(f"censoring type was {censoring_type} but can only be "
                     "lower or upper")

  flat_table.loc[:, censoring_type] =\
    [fun(x, y + '_' + z.lower()) for
     x,y,z in zip(flat_table.loc[:,"titre"], flat_table.loc[:,"lab_code"],
                  flat_table.loc[:,"assay_type"])]



def lower(titre, lab_code):
  '''
  based on the dictionary _code_to_LLD, returns a lower threshold value
  for the given lab_code. It also does some checks based on the given value.

  few labs have reported multiple lower thresholds which is stored as a list
  if titre does not have < in it then lowest of these is taken. if titre
  has < in it then it is checked that this value should exist in this list
  and threshold is take as this value.
  '''

  lld = _code_to_LLD[lab_code]

  if not isinstance(lld, list): #there are labs with multiple llds...
    lld = [lld]

  lld = np.array(lld)

  log_lld = np.array([log(x) for x in lld])
  log_titre = log(titre)

  if isinstance(titre, float) and np.isnan(titre):
    return np.min(log_lld)

  if (str(titre)[0]=='<' and log_titre not in log_lld) or\
    (str(titre)[0]!='<' and log_titre < np.min(log_lld)):
      raise ValueError(f"lab:{lab_code}, titre: {titre}, lld: {lld}")

  if str(titre)[0]=='<':
    return log_titre


  return np.min(log_lld)


def upper(titre, lab_code):
  '''
  similar to lower above but for upper thresholding
  '''

  uld = _code_to_ULD[lab_code]
  log_uld = log(uld)
  log_titre = log(titre)

  if isinstance(titre, float) and np.isnan(titre):
    return log_uld

  if (str(titre)[0]=='>' and log_titre != log_uld) or\
    (str(titre)[0]!='>' and log_titre > log_uld):
      raise ValueError(f"lab:{lab_code}, titre: {titre}, uld: {uld}")

  return log_uld
