#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:31:37 2024

@author: avicenna
"""

# pylint: disable=bad-indentation, wrong-import-position, import-error


import tempfile
import os
import subprocess
from os import listdir
from os.path import isfile, join

import six

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

cdir = os.path.dirname(os.path.realpath(__file__))

antigens = ["Alpha", "WT", "Beta", "Delta", "BA.1", "BA.5", "XBB.1.5"]

default_aa_colors={
  "G":[0.97,0.8,0.57,1],
  "A":[0.97,0.8,0.57,1],
  "S":[0.97,0.8,0.57,1],
  "T":[0.97,0.8,0.57,1],
  "C":[0.62,0.98,0.59,1],
  "V":[0.62,0.98,0.59,1],
  "I":[0.62,0.98,0.59,1],
  "L":[0.62,0.98,0.59,1],
  "P":[0.62,0.98,0.59,1],
  "F":[0.62,0.98,0.59,1],
  "Y":[0.62,0.98,0.59,1],
  "M":[0.62,0.98,0.59,1],
  "W":[0.62,0.98,0.59,1],
  "N":[0.99,0.57,0.99,1],
  "Q":[0.99,0.57,0.99,1],
  "H":[0.99,0.57,0.99,1],
  "Z":[0.99,0.57,0.99,1],
  "D":[0.99,0.57,0.55,1],
  "E":[0.99,0.57,0.55,1],
  "B":[0.99,0.57,0.55,1],
  "K":[0.5,0.5,1,1],
  "R":[0.5,0.5,1,1],
  "*":[1/3,1/3,1/3,1],
  "X":[1/3,1/3,1/3,1],
  ".":[0.73,0.73,0.73,1],
  "":[1/3,1/3,1/3,1],
  "-":[0.56,0.56,0.56,1],

  }


def sequence_plot(seqs, positions, alpha=0.5, fontsize=14):

  '''
  Given a list of Bio SequenceRecord objects and a subset of positions
  this function produces a sequnce plot in the shape of a table
  '''

  assert all(len(seq)>max(positions) for seq in seqs)
  assert all(p>=0 for p in positions)

  seq_lists = [list((str(seq.seq))) for seq in seqs]
  seq_lists = [[seq_list[p] for p in positions] for seq_list in seq_lists]

  sequence_table = pd.DataFrame(seq_lists,
                                index=[x.id.replace('_',' ') for x in seqs],
                                columns=[x+1 for x in positions],
                                dtype=str)
  aa_colors = default_aa_colors.copy()
  for aa in aa_colors:
    aa_colors[aa][-1] = alpha

  wt_seq = sequence_table.iloc[0, :].values
  cell_colors = []
  for ind_label,label in enumerate(sequence_table.index):

    seq = sequence_table.loc[label, :].values

    if ind_label != 0:
      cell_colors.append([aa_colors[x] if x != wt_seq[ind] else
                          aa_colors['.'] for ind,x in enumerate(seq)])
    else:
      cell_colors.append([aa_colors[x] for ind,x in enumerate(seq)])

  s1,s2 = sequence_table.shape

  colors = [[0.9,0.9,0.9] if i%2==0 else [1,1,1] for i in range(s1)]

  for i in range(1,s1):
    for j in range(s2):
        if sequence_table.iloc[i,j] == wt_seq[j]:
            sequence_table.iloc[i,j] = '.'


  rh = max(len(str(p)) for p in positions)/4
  rw = max(len(label) for label in sequence_table.index)/15

  fig, ax = plt.subplots(1, 1, figsize=(len(positions)*0.4 + 0.1*rw,
                                        (len(seqs)+1.5*rh)*0.5))
  fig.patch.set_visible(False)
  ax.axis('off')
  ax.axis('tight')

  header_height = 1.5*rh/len(seqs)


  sequence_table.index = [x.replace('_','+') for x in sequence_table.index]

  table = ax.table(cellText=sequence_table.values,
                   colLabels=sequence_table.columns,
                   rowLoc='left', cellLoc='center',
                   loc='center', rowLabels = sequence_table.index,
                   colLoc='center', cellColours=cell_colors,
                   rowColours = colors , alpha=alpha)

  for _, cell in six.iteritems(table._cells):
      cell.set_edgecolor([0.85,0.87,0.88])
      cell.set_linewidth(3)

  table.auto_set_font_size(False)
  table.set_fontsize(fontsize)

  for i in range(len(positions)):

    cell = table._cells[0,i]
    cell.get_text().set_rotation(-90)
    cell.set_height(header_height)
    cell.set_text_props(ha="left")
    cell.set_text_props(fontproperties=FontProperties(size=fontsize,
                                                      family="Monospace"))


  for i in range(-1,len(positions)):
    for j in range(1,len(sequence_table.index)+1):
      cell = table._cells[j,i]
      table._cells[j,i].set_height((1-header_height)/len(seqs))
      cell.set_text_props(fontproperties=FontProperties(size=fontsize,
                                                        family="Monospace"))
      #cell.set_text_props(ha="center",va="center")

  return fig,ax


def mafft_align(seqs, output_path=None, ids=None, file_type='fasta', nthreads=1,
                alignment_method='auto', extra_args=None, verbose=True):

  '''
  given a list of seqs in str format, does multiple alignment using
  MAFFTS calling it with subprocess
  '''

  if ids is None:
      ids = [f'Seq{i}' for i in range(len(seqs))]
  else:
      assert len(ids) == len(seqs)

  if extra_args is None:
    extra_args = ''

  if alignment_method == 'local_accurate':
    args = '--localpair --maxiterate 1000 ' + extra_args

  elif alignment_method == 'global_accurate':
    args = '--globalpair --maxiterate 1000 ' + extra_args

  elif alignment_method == 'unalignable_accurate':
    args = '--ep 0 --genafpair --maxiterate 1000 ' + extra_args

  elif alignment_method == 'fast1':
    args = '--retree 2 --maxiterate 2 ' + extra_args

  elif alignment_method == 'fast2':
    args = '--retree 2 --maxiterate 1000 ' + extra_args

  elif alignment_method == '2000+':
    args = '--retree 1 --maxiterate 0 ' + extra_args

  elif alignment_method == '10000+':
    args = '--retree 1 --maxiterate 0 --nofft --parttree ' + extra_args

  elif alignment_method == 'auto':
    args = '--auto '

  else:
      raise ValueError(f'{alignment_method} is not a known alignment method')

  records = []

  for sid,seq in zip(ids,seqs):
    record = SeqRecord(Seq(seq),id=sid)
    records.append(record)

  with tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=False) as tmp:
    tmp_path = tmp.name
    write_seqs(records, tmp_path, file_type)

  if output_path is None:
    with tempfile.NamedTemporaryFile(suffix=f'.{file_type}') as output_tmp:
      output_path = output_tmp.name

  cmd = f'mafft {args} --thread {nthreads} {tmp_path}> {output_path}'

  if not verbose:
    cmd += " 2>/dev/null"

  subprocess.call(cmd, shell=True)

  return read_seqs(output_path)



def read_seqs(input_path, file_type = 'fasta'):

  '''
  convenience wrapper for SeqIO parse
  '''
  return SeqIO.parse(input_path, file_type)


def write_seqs(records, output_path, file_type='fasta'):
  '''
  Writes sequences to file. if type is not txt uses SeqIO.write,
  if txt then writes it manually.
  '''

  if file_type != 'txt':
    SeqIO.write(records, output_path, file_type)
  else:
    with open(output_path,'w', encoding="utf-8") as fp:
      rec_ids = []
      seqs = []

      for record in records:
        rec_ids.append(record.id)
        seqs.append(str(record.seq).replace('\n',''))

      max_len = max(len(x) for x in rec_ids)

      for rec_id,seq in zip(rec_ids,seqs):
        print(rec_id)
        fp.write(f'{rec_id.ljust(max_len+1)} : {seq}\n')


def translate(nt_seq, verbose=False):
  '''
  given nt_seq, translate to aa using all frames and take the one with least
  number of stop codons
  '''

  stop_counts = []
  for i in range(3):
    aa_seq = str(nt_seq[i:].translate().seq)
    stop_counts.append(aa_seq.count('*'))

  I = np.argmin(stop_counts)
  if verbose:
    print(f"{stop_counts} {I}")

  return nt_seq[I:].translate()


def _count_head(seq):

  if seq[0] != '-':
    return 0

  I = [inda for inda, a in enumerate(seq[:-1]) if a=='-' and seq[inda+1] !='-']

  return I[0] + 1


def find_variant_positions(output):
  '''
  given a list of aligned sequences (in SeqRecord format)
  finds positions for which there is more than one type of letter
  '''

  seqs = [list(str(seq.seq)) for seq in output]

  heads = [_count_head(seq.seq) for seq in output]
  tails = [len(seq.seq) - _count_head(seq.seq[::-1]) for seq in output]

  start = max(heads)
  end = min(tails)

  seqs = [seq[start:end] for seq in seqs]

  aas = [set(seq[i] for seq in seqs).difference(['X']) for i in range(len(seqs[0]))]

  return [ind for ind in range(len(aas)) if len(aas[ind])>1]


def _replace_name(name):

  return name.replace("BA1","BA.1").replace("BA5","BA.5").replace("XBB","XBB.1.5").replace('_',' ')



def _main():

  seq_dir = f"{cdir}/../data/sequences"

  i0 = 4999
  i1 = 8413

  names = [f.split('.')[0].split("_EPI")[0] for f in listdir(seq_dir) if isfile(join(seq_dir, f))]
  paths = [join(seq_dir, f) for f in listdir(seq_dir) if isfile(join(seq_dir, f))]

  I = sorted(range(len(names)),
             key = lambda i: [a.replace('.','') not in
                              names[i] for a in antigens]+
                             [ "WHO" not in names[i] for a in antigens])


  col1 = [_replace_name(x.split('_EPI_ISL_')[0].split('/')[-1]) for x in paths]
  col2 = ['EPI_ISL_' + x.split('_EPI_ISL_')[1].replace('.fasta','')
          for x in paths]

  acc_table = pd.DataFrame(np.array([col1,col2]).T,
                           columns=["Variant", "Accession Number"],
                           dtype=str)
  acc_table = acc_table.iloc[I,:]
  acc_table.to_csv(f"{cdir}/outputs/accession_numbers.csv", header=True, index=False)

  aa_seqs = [translate(list(read_seqs(path))[0], True).seq[i0:i1]
             for path in paths]

  output = list(mafft_align(list(aa_seqs), ids=names))
  output = [seq for seq in output if seq.id != "ref"]
  output = [output[i] for i in I]

  heads = [_count_head(seq.seq) for seq in output]
  start = max(heads[1:]) + 2040 - 90
  end = 3345

  output = [seq[start:end] for seq in output]
  subset_positions = find_variant_positions(output)

  for _,o in enumerate(output):
    o.id = _replace_name(o.id)

  fig, _ = sequence_plot(output, sorted(subset_positions),
                          fontsize=16)
  fig.tight_layout()
  fig.savefig(f"{cdir}/plots/sequences_plot.png")
  plt.close("all")

if __name__ == "__main__":

  _main()
