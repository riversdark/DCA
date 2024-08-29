# Description
This is a PyTorch implementation of Potts models for Direct Coupling Analysis (DCA). 
Given a Multiple Sequence Alignment (MSA) for a protein, DCA is aimed to calculate the direct coupling between pairs of positions.
It can be used to predict the contact map of proteins using MSA and the predicted contact map is useful for 
predicting protein 3D structures.

One effective method for DCA is using Potts models, which are also called Boltzmann machines in machine learning.
Potts models belong to a larger category of models called generative probabilistic models, which means the model 
assigns a probablity for each sample and we can generate new samples by sampling from this distribution.
The first step when using these generative probabilistic models is to learn a model from observed data.
In the context of DCA, the observed data is a MSA and each sequence from the MSA is one sample.
As the Potts model assign probabilities for all the samples, theretically, we can write the likelihood
function of the data and use Maximum Likelihood Estimation (MLE) to learn the model.
In practice, the MLE method does not work well because it requires calcualting or estimating
the normalization constant of the model distribution. Several approximation methods have been developed
to do the learning, such as Pseudo Maximum Likelihood Method, Score Matching, and Adaptive Cluter Expansion method among others.

In this implementation, the pseudo maximum likelihood method is used for learning the model. L-BFGS mehod is used for optimization and 
in the optimization, the weight matrix of the Potts model is restrainted to be symmetric. L2-normed penalty on the weight parameters is used for regulization.

# Reference
The implementation here mainly follows the method presented in reference 1. I also list the reference 2 and 3, which introduced pseduo maximum likelihood method and average-product correction method, respectively.

1. Ekeberg, Magnus, et al. "Improved contact prediction in proteins: using pseudolikelihoods to infer Potts models." Physical Review E 87.1 (2013): 012707.
2. Besag, Julian. "Statistical analysis of non-lattice data." The statistician (1975): 179-195.
3. Dunn, Stanley D., Lindi M. Wahl, and Gregory B. Gloor. "Mutual information without the influence of phylogeny or entropy dramatically improves residue contact prediction." Bioinformatics 24.3 (2007): 333-340.

# Example
In this example, we first download a MSA from Pfam and use the MSA to train a Potts model. You can also use other method to make a MSA. Based on the trained Potts model, we calculate interaction scores between pairs of positions using average-product correction (AFC) method. At the end, we compare the pairs of postions with high interaction scores with native contact map obtained from a PDB structure.

1. **Download a MSA from Pfam.**

   Pfam is a database of protein families that includes their annotations and multiple sequence alignments. Each Pfam entry represents a protein family or domain and is assigned a unique identifier.

   In this example, we use PF00041, which represents the Fibronectin type III domain. This domain is found in a wide variety of extracellular proteins and is involved in cell surface binding. It's approximately 100 amino acids long and has a beta sandwich structure.

   Given a Pfam ID (PF00041), `./script/download_MSA.py` downloads the corresponding multiple sequence alignment and saves it in the file `./pfam_msa/PF00041_full.txt`
   ```
   python ./script/download_MSA.py --Pfam_id PF00041
   ```

2. **Process the MSA.**

   The downloaded MSA cannot be used directly to train the Potts model. It has to be processed into a specific format using `./script/process_MSA.py`. A query sequence is used as the reference sequence to clean up the MSA. The results are saved in directory `./pfam_msa/`.

   ```
   python ./script/process_MSA.py --msa_file ./pfam_msa/PF00041_full.txt --query_id A0A7K4UZQ6_9EMBE/901-989 --output_dir ./pfam_msa/
   ```

   In this command:
   - `--msa_file`: Specifies the path to the downloaded MSA file.
   - `--query_id`: Specifies the reference sequence ID used to clean up the MSA. In this case, it's `A0A7K4UZQ6_9EMBE/901-989`, where 901-989 indicates the range of amino acid positions included in the alignment.
   - `--output_dir`: Specifies where the processed files will be saved.

   The script processes the MSA and generates several output files:
   - `aa_index.pkl`: A dictionary mapping amino acids to numerical indices.
     Dimension: 21 entries (20 amino acids + gap)
   - `seq_pos_idx.pkl`: A list of indices representing positions kept after filtering.
     Dimension: [L], where L is the number of positions kept
   - `seq_msa.pkl`: The processed MSA as a numerical array.
     Dimension: [N, L], where N is the number of sequences and L is the sequence length
   - `seq_weights.pkl`: An array of sequence weights based on amino acid frequencies.
     Dimension: [N], where N is the number of sequences
   - `seq_msa_binary.pkl`: A binary representation of the MSA.
     Dimension: [N, L, 21], where N is the number of sequences, L is the sequence length, and 21 represents the one-hot encoding of amino acids (20) plus gap

   These files are used in subsequent steps for training the Potts model.

   TODO: Investigate the query ID issue, as the MSA content may change over time.

3. **Learn the Potts model.**

   Here we set the hyperparameters for learning the Potts model: 200 for maximum number of optimization steps, 0.05 for weight decay factor, and a batch size of 500. The resulting Potts model is saved as `./model/model_weight_decay_0.050.pkl`.

   ```
   python ./script/Potts_model.py --input_dir ./pfam_msa/ --max_iter 200 --weight_decay 0.05 --output_dir ./model/ --batch_size 500
   ```

   The Potts model is a statistical physics model used in Direct Coupling Analysis (DCA) to capture co-evolutionary relationships between amino acid positions in a protein sequence. It models the probability distribution of sequences in the Multiple Sequence Alignment (MSA).

   Key components of the model:

   - **Inputs**:
     - `seq_msa_binary`: Binary representation of the MSA [N, L * K]
     - `seq_weight`: Weights for each sequence [N]
     - `seq_msa_idx`: Integer representation of the MSA [N, L]

   - **Parameters**:
     - `J`: Coupling matrix [L * K, L * K]
     - `h`: Fields [L * K]
     - `J_mask`: Binary mask for J [L * K, L * K]

   Where N is the number of sequences, L is the sequence length, and K is the number of amino acid types (usually 21, including gaps).

   The model is trained using batched processing to handle large datasets efficiently. For each batch:

   1. Compute logits: `logits = torch.matmul(batch_seq, J*J_mask) + h`
   2. Calculate cross-entropy loss
   3. Add L2 regularization: `weight_decay * torch.sum((J*J_mask)**2)`
   4. Compute gradients and accumulate them

   After training, the learned parameters `J` and `h` can be used to predict contacts between residues in the protein structure, generate new sequences, or assess the probability of sequences under the model.

   Note: The current implementation accumulates gradients across all batches before updating parameters. This approach may be memory-intensive for very large datasets.

   TODO: Investigate GPU usage optimization, as current batch size doesn't seem to affect GPU utilization.

4. **Calculate and plot the interaction score.**

   ```
   python ./script/calc_score.py --model_file ./model/model_weight_decay_0.050.pkl --N 80
   ```
   Given the model `./model/model_weight_decay_0.050.pkl`, `./script/calc_score.py` is used to calculate the interaction scores
   between pairs of positions in the MSA. Based on the predicted interaction scores, the top 80 pairs of positions are predicted
   to have contacts and are plotted in the following figure. Also plotted in the following figure are the native contacts obtained
   from a PDB structure.

   The interaction score calculation process involves the following steps:
   - **Extract Submatrices**: For each pair of positions, a submatrix \( J' \) is extracted from the coupling matrix \( J \).
   - **Mean-Centering**: The submatrix \( J' \) is mean-centered by subtracting the mean of each row and column, and then adding the overall mean.
   - **Frobenius Norm Score**: The Frobenius norm of the mean-centered submatrix \( J' \) is calculated to obtain the interaction score.
   - **Score Correction**: The scores are corrected by removing the mean interaction score for each position.
   - **Thresholding**: Scores for positions that are too close (within 4 positions) are set to \(-\infty\). The top 80 pairs of positions with the highest scores are selected as predicted contacts.

   ![Figure](./output/contact_potts.png)
