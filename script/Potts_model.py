__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2017/05/13 23:43:38"

import numpy as np
import pickle
import torch
import torch.nn as nn
from scipy import optimize
from sys import exit
import sys
import timeit
import argparse
import subprocess
import torch.cuda

parser = argparse.ArgumentParser(description = "Learn a Potts model using Multiple Sequence Alignment data.")
parser.add_argument("--input_dir",
                    help = "input directory where the files seq_msa_binary.pkl, seq_msa.pkl, seq_weight.pkl are.")
parser.add_argument("--max_iter",
                    help = "The maximum num of iteratioins in L-BFGS optimization.",
                    type = int)
parser.add_argument("--weight_decay",
                    help = "weight decay factor of L2 penalty",
                    type = float)
parser.add_argument("--output_dir",
                    help = "output directory for saving the model")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU")

args = parser.parse_args()

## read msa
msa_file_name = args.input_dir + "/seq_msa_binary.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa_binary = pickle.load(input_file_handle)
seq_msa_binary = seq_msa_binary.astype(np.float32)

msa_file_name = args.input_dir + "/seq_msa.pkl"
with open(msa_file_name, 'rb') as input_file_handle:
    seq_msa = pickle.load(input_file_handle)
seq_msa = seq_msa.astype(np.float32)

weight_file_name = args.input_dir + "/seq_weight.pkl"
with open(weight_file_name, 'rb') as input_file_handle:
    seq_weight = pickle.load(input_file_handle)
seq_weight = seq_weight.astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
print(f"Using device: {device}")

seq_msa_binary = torch.from_numpy(seq_msa_binary).to(device)
seq_weight = torch.from_numpy(seq_weight).to(device)
weight_decay = float(args.weight_decay)

## pseudolikelihood method for Potts model
_, len_seq, K = seq_msa_binary.shape
num_node = len_seq * K

seq_msa_binary = seq_msa_binary.reshape(-1, num_node)  # dim: [num_sequences, num_node]
seq_msa_idx = torch.argmax(seq_msa_binary.reshape(-1, K), -1).reshape(-1, len_seq)  # dim: [num_sequences, len_seq]

# h = seq_msa_binary.new_zeros(num_node, requires_grad = True)
# J = seq_msa_binary.new_zeros((num_node, num_node), requires_grad = True)
J_mask = seq_msa_binary.new_ones((num_node, num_node))
for i in range(len_seq):
    J_mask[K*i:K*i+K, K*i:K*i+K] = 0


def calc_loss_and_grad(parameter):
    parameter = parameter.astype(np.float32)
    J = parameter[0:num_node**2].reshape([num_node, num_node])
    h = parameter[num_node**2:]
    
    J = torch.tensor(J, requires_grad=True, device=device)  # dim: [num_node, num_node]
    h = torch.tensor(h, requires_grad=True, device=device)  # dim: [num_node]
    
    batch_size = args.batch_size
    num_batches = (seq_msa_binary.shape[0] + batch_size - 1) // batch_size
    
    total_loss = 0
    total_grad_J = torch.zeros_like(J)
    total_grad_h = torch.zeros_like(h)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, seq_msa_binary.shape[0])
        
        batch_seq = seq_msa_binary[start_idx:end_idx]  # dim: [batch_size, num_node]
        batch_weight = seq_weight[start_idx:end_idx]  # dim: [batch_size]
        
        # Calculate energy for each sequence
        logits = torch.matmul(batch_seq, J*J_mask) + h  # dim: [batch_size, num_node]
        energy = torch.sum(batch_seq * logits, dim=1)  # dim: [batch_size]
        
        # Calculate partition function (normalization constant)
        all_states = torch.eye(K, device=device).repeat(len_seq, 1)  # dim: [num_node, K]
        all_logits = torch.matmul(all_states, J*J_mask) + h  # dim: [num_node, K]
        all_energies = torch.sum(all_states * all_logits, dim=1)  # dim: [num_node]
        Z = torch.sum(torch.exp(-all_energies.reshape(len_seq, K)), dim=1)  # dim: [len_seq]
        log_Z = torch.sum(torch.log(Z))  # scalar
        
        # Calculate full likelihood
        log_likelihood = -energy - log_Z  # dim: [batch_size]
        batch_loss = -torch.sum(log_likelihood * batch_weight)  # scalar
        batch_loss += weight_decay * torch.sum((J*J_mask)**2)  # weight decay
        
        batch_loss.backward()

        total_loss += batch_loss.item()
        total_grad_J += J.grad
        total_grad_h += h.grad
        
        J.grad.zero_()
        h.grad.zero_()
    
    grad_J = total_grad_J.cpu().numpy()
    grad_h = total_grad_h.cpu().numpy()
    
    grad = np.concatenate((grad_J.reshape(-1), grad_h))
    grad = grad.astype(np.float64)
    return total_loss, grad

init_param = np.zeros(num_node*num_node + num_node) # dubious? terrible?
#loss, grad = calc_loss_and_grad(init_param)
param, f, d = optimize.fmin_l_bfgs_b(calc_loss_and_grad, init_param, iprint = True, maxiter = args.max_iter)
J = param[0:num_node**2].reshape([num_node, num_node])
h = param[num_node**2:]

## save J and h
model = {}
model['len_seq'] = len_seq
model['K'] = K
model['num_node'] = num_node
model['weight_decay'] = args.weight_decay
model['max_iter'] = args.max_iter
model['J'] = J
model['h'] = h

subprocess.run(['mkdir', '-p', args.output_dir])
with open("{}/model_weight_decay_{:.3f}.pkl".format(args.output_dir, args.weight_decay), 'wb') as output_file_handle:
    pickle.dump(model, output_file_handle)
