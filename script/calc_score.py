__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/15 01:42:31"

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', size = 16)
mpl.rc('axes', titlesize = 'large', labelsize = 'large')
mpl.rc('xtick', labelsize = 'large')
mpl.rc('ytick', labelsize = 'large')
import os
import sys
import argparse
import subprocess

parser = argparse.ArgumentParser(description = "Calculate pairwise interaction scores infered from a Potts model and plot the top N interactions.")
parser.add_argument('--model_file',
                    help = "path to a Potts model file")
parser.add_argument('--N',
                    help = "the top N scored interactions are plotted",
                    type = int)
args = parser.parse_args()

def calculate_contact_map(model, N):
    len_seq = model['len_seq']
    K = model['K']
    J = model['J']

    ## calculate interaction scores
    J_prime_dict = {}
    score_FN = np.zeros([len_seq, len_seq])
    for i in range(len_seq):
        for j in range(i+1, len_seq):
            J_prime = J[(i*K):(i*K+K), (j*K):(j*K+K)]
            J_prime = J_prime - J_prime.mean(0).reshape([1,-1]) - J_prime.mean(1).reshape([-1,1]) + J_prime.mean()
            J_prime_dict[(i,j)] = J_prime
            score_FN[i,j] = np.sqrt(np.sum(J_prime * J_prime))
            score_FN[j,i] = score_FN[i,j]
    score_CN = score_FN - score_FN.mean(1).reshape([-1,1]).dot(score_FN.mean(0).reshape([1,-1])) / np.mean(score_FN)

    for i in range(score_CN.shape[0]):
        for j in range(score_CN.shape[1]):
            if abs(i-j) <= 4:
                score_CN[i,j] = -np.inf
            
    tmp = np.copy(score_CN).reshape([-1])
    tmp.sort()
    cutoff = tmp[-N*2]
    contact_plm = score_CN > cutoff
    for j in range(contact_plm.shape[0]):
        for i in range(j, contact_plm.shape[1]):
            contact_plm[i,j] = False
    
    return contact_plm

## load model
with open(args.model_file, 'rb') as input_file_handle:
    model = pickle.load(input_file_handle)

contact_plm = calculate_contact_map(model, args.N)


######################################################################
#### Contact Maps from PDB Structure ######
######################################################################
def calculate_contact_map_pdb(protein, distance_pdb_file, seq_pos_idx_file, cutoff=8.5, offset=0):
    dist = []
    with open(distance_pdb_file, 'r') as f:
        for l in f.readlines():
            l = l.strip().split(",")
            dist.append([int(l[0]), int(l[1]), float(l[2])])
    
    id_pdb = sorted(set([a[0] for a in dist] + [b[1] for b in dist]))
    num_res = len(id_pdb)

    dist_array = np.zeros((num_res, num_res))
    for d in dist:
        i, j = id_pdb.index(d[0]), id_pdb.index(d[1])
        dist_array[i,j] = dist_array[j,i] = d[2]
    
    contact_pdb = dist_array <= cutoff
    for i in range(contact_pdb.shape[0]):
        for j in range(contact_pdb.shape[1]):
            if abs(id_pdb[i] - id_pdb[j]) <= 4:
                contact_pdb[i,j] = False

    with open(seq_pos_idx_file, 'rb') as f:
        position_idx = np.array(pickle.load(f))
    
    contact_pdb = contact_pdb[position_idx + offset,:][:,position_idx + offset]

    for j in range(contact_pdb.shape[0]):
        for i in range(j, contact_pdb.shape[1]):
            contact_pdb[i,j] = False
    
    return contact_pdb

protein = "Fibronectin_III"
distance_pdb_file = "./pdb/dist_pdb.txt"
seq_pos_idx_file = "./pfam_msa/seq_pos_idx.pkl"
contact_pdb = calculate_contact_map_pdb(protein, distance_pdb_file, seq_pos_idx_file)

def plot_contacts(contact_pdb, contact_plm, protein, output_path):
    fig = plt.figure(figsize=(10, 10))
    fig.clf()

    # Plot native contacts from PDB
    I, J = np.where(contact_pdb)
    if len(I) > 0 and len(J) > 0:
        plt.plot(I, J, 'bo', alpha=0.2, markersize=8, label='native contacts from PDB')

    # Plot predicted contacts from Potts model
    I, J = np.where(contact_plm)
    if len(I) > 0 and len(J) > 0:
        plt.plot(I, J, 'r^', markersize=6, mew=1.5, label='predicted contacts from Potts model')

    # Set aspect ratio and axis limits
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, contact_pdb.shape[0])
    plt.ylim(0, contact_pdb.shape[1])

    # Set title and legend
    plt.title(protein)
    plt.legend()

    # Create output directory and save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

# Call the function
output_path = "./output/contact_comparison.png"
plot_contacts(contact_pdb, contact_plm, protein, output_path)
sys.exit()
