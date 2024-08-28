__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/10 19:51:12"

import pickle
import sys
import numpy as np
from sys import exit
import argparse

parser = argparse.ArgumentParser(description = "Process a Stockholm formatted multiple sequence alignment into matrix, which can be used for building a Potts model.")
parser.add_argument('--msa_file', help = "path to the multiple sequence alignment file")
parser.add_argument('--query_id', help = "one query id from the msa_file")
parser.add_argument('--output_dir', help = "output directory")
args = parser.parse_args()

#query_seq_id = "TENA_HUMAN/804-884"
## read all the sequences into a dictionary
seq_dict = {}
with open(args.msa_file, 'r') as file_handle:
    for line in file_handle:
        line = line.strip()        
        if line == "" or line[0] == "#" or line[0] == "/" or line[0] == "":
            continue
        seq_id, seq = line.split()
        seq_dict[seq_id] = seq.upper()

# Add debugging information
print(f"Total sequences read: {len(seq_dict)}")
print(f"First 5 sequence IDs: {list(seq_dict.keys())[:5]}")

# Check if the query_id exists, if not, try to find a close match
if args.query_id not in seq_dict:
    print(f"Warning: Query ID '{args.query_id}' not found in the sequence dictionary.")
    close_matches = [key for key in seq_dict.keys() if args.query_id.split('/')[0] in key]
    if close_matches:
        print(f"Found close matches: {close_matches}")
        args.query_id = close_matches[0]
        print(f"Using '{args.query_id}' as the query ID.")
    else:
        print("No close matches found. Please check your query ID and MSA file.")
        print(f"Available IDs: {list(seq_dict.keys())[:10]}...")  # Print first 10 IDs
        sys.exit(1)

## remove gaps in the query sequences
query_seq = seq_dict[args.query_id] ## with gaps
idx = [ s == "-" or s == "." for s in query_seq]
for k in seq_dict.keys():
    seq_dict[k] = [seq_dict[k][i] for i in range(len(seq_dict[k])) if idx[i] == False]
query_seq = seq_dict[args.query_id] ## without gaps

## remove sequences with too many gaps
len_query_seq = len(query_seq)
seq_id = list(seq_dict.keys())
for k in seq_id:
    if seq_dict[k].count("-") + seq_dict[k].count(".") >= len_query_seq * 0.20:
        seq_dict.pop(k)
        
## convert aa type into num 0-20
aa = ['R', 'H', 'K',
      'D', 'E',
      'S', 'T', 'N', 'Q',
      'C', 'G', 'P',
      'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
aa_index = {}
aa_index['-'] = 0
aa_index['.'] = 0
i = 1
for a in aa:
    aa_index[a] = i
    i += 1
# Add 'B' (Aspartic acid or Asparagine) and 'Z' (Glutamic acid or Glutamine) to aa_index
aa_index['B'] = aa_index['D']  # Treat 'B' as Aspartic acid
aa_index['Z'] = aa_index['E']  # Treat 'Z' as Glutamic acid
aa_index['X'] = 0  # Treat 'X' (any amino acid) as a gap

with open(args.output_dir + "/aa_index.pkl", 'wb') as file_handle:
    pickle.dump(aa_index, file_handle)

seq_msa = []
for k in seq_dict.keys():
    seq_msa.append([aa_index.get(s, 0) for s in seq_dict[k]])  # Use .get() with default 0
seq_msa = np.array(seq_msa)

## remove positions where too many sequences have gaps
pos_idx = []
for i in range(seq_msa.shape[1]):
    if np.sum(seq_msa[:,i] == 0) <= seq_msa.shape[0]*0.2:
        pos_idx.append(i)
with open(args.output_dir + "/seq_pos_idx.pkl", 'wb') as file_handle:
    pickle.dump(pos_idx, file_handle)
    
seq_msa = seq_msa[:, np.array(pos_idx)]
with open(args.output_dir + "/seq_msa.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa, file_handle)


## reweighting sequences
seq_weight = np.zeros(seq_msa.shape)
for j in range(seq_msa.shape[1]):
    aa_type, aa_counts = np.unique(seq_msa[:,j], return_counts = True)
    num_type = len(aa_type)
    aa_dict = {}
    for a in aa_type:
        aa_dict[a] = aa_counts[list(aa_type).index(a)]
    for i in range(seq_msa.shape[0]):
        seq_weight[i,j] = (1.0/num_type) * (1.0/aa_dict[seq_msa[i,j]])
tot_weight = np.sum(seq_weight)
seq_weight = seq_weight.sum(1) / tot_weight 
with open(args.output_dir + "/seq_weight.pkl", 'wb') as file_handle:
    pickle.dump(seq_weight, file_handle)

## change aa numbering into binary
K = 21 ## num of classes of aa
D = np.identity(K)
num_seq = seq_msa.shape[0]
len_seq_msa = seq_msa.shape[1]
seq_msa_binary = np.zeros((num_seq, len_seq_msa, K))
for i in range(num_seq):
    seq_msa_binary[i,:,:] = D[seq_msa[i]]

with open(args.output_dir + "/seq_msa_binary.pkl", 'wb') as file_handle:
    pickle.dump(seq_msa_binary, file_handle)
