import numpy as np
import bnpy

from bnpy.allocmodel.hmm.HMMUtil import runViterbiAlg
from bnpy.util.StateSeqUtil import alignEstimatedStateSeqToTruth
from bnpy.util.StateSeqUtil import calcHammingDistance
from scipy import stats
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 8)
plt.rcParams['font.size'] = 20

# Plot Hamming distance
def plot_hamming(experiment_out, title=None):
    L_vals = []
    sparse_ham = []
    blocked_ham = []
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for model, info_dict, label in experiment_out:
        alg_dict = info_dict['KwArgs']['VB']
        L = alg_dict['nnzPerRowLP']
        blocked = alg_dict['blockedLP']
        
        ham_dist_history = compute_hamming(info_dict)
        ham_dist = ham_dist_history[-1]
        
        # Plot Hamming distance vs iteration
        marker = 'o' if blocked else 'x'
        color = colors[L%5] if L > 0 else 'yellow'
        plt.plot(info_dict['lap_history'], ham_dist_history,
                 label=label, marker=marker, color=color)

        # Gather info for plotting loss vs L
        if L == 0 or L == 1:
            L_tmp = L if L == 1 else info_dict['K_history'][-1]
            L_vals.append(L_tmp)

            blocked_ham.append(ham_dist)
            sparse_ham.append(ham_dist)
        elif alg_dict['blockedLP']:
            L_vals.append(L)
            blocked_ham.append(ham_dist)
        else:
            sparse_ham.append(ham_dist)

    plt.xlabel('iteration')
    plt.ylabel('Hamming distance')
    plt.legend()
    if title: plt.title(title)
    plt.show()

    # Plot Hamming distance vs L
    plt.plot(L_vals[:-1], sparse_ham[:-1], label='One-pass sparse', marker='o')
    plt.plot(L_vals[:-1], blocked_ham[:-1], label='Two-pass sparse', marker='o')
    plt.hlines(sparse_ham[-1], L_vals[0], L_vals[-2], label='Dense model')
    plt.xlabel('L')
    plt.ylabel('Hamming distance')
    plt.legend()
    if title: plt.title(title)
    plt.show()

def compute_hamming(model_dict):
    dataset = model_dict['Data']
    N = dataset.X.shape[0]
    doc_range = dataset.doc_range
    ztrue = dataset.TrueParams['Z']
    path = model_dict['task_output_path']
    laps = np.concatenate(([0], model_dict['lap_history']))
    ham_dist_history = np.empty_like(model_dict['lap_history'], dtype=float)
    
    # Iterate over laps
    for lap in laps:
        # Load model
        model, _ = bnpy.load_model_at_lap(path, lap)
    
        # Load model parameters
        init = model.allocModel.get_init_prob_vector()
        trans = model.allocModel.get_trans_prob_matrix()
        evidence = model.obsModel.calcLogSoftEvMatrix_FromPost(dataset)
        
        # Run Viterbi
        zhat = np.empty(N)
        for i, (start, end) in enumerate(zip(doc_range[:-1], doc_range[1:])):
            zhat[start:end] = runViterbiAlg(evidence[start:end],
                                            np.log(init), np.log(trans))
        zhat_aligned = alignEstimatedStateSeqToTruth(zhat, ztrue)

        # Compute Hamming distance
        ham_dist_history[lap - 1] = calcHammingDistance(ztrue, zhat_aligned)
        
    return ham_dist_history

# Plot loss
def plot_loss(experiment_out, title=None):    
    # Plot loss vs iteration, and save info for plotting loss vs L
    L_vals = []
    sparse_loss = []
    blocked_loss = []
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for model, info_dict, label in experiment_out:
        loss = info_dict['loss']
        alg_dict = info_dict['KwArgs']['VB']
        L = alg_dict['nnzPerRowLP']
        blocked = alg_dict['blockedLP']
        
        # Plot loss vs iteration
        marker = 'o' if blocked else 'x'
        color = colors[L%5] if L > 0 else 'yellow'
        plt.plot(info_dict['lap_history'], info_dict['loss_history'],
                 label=label, marker=marker, color=color)
        
        # Gather info for plotting loss vs L
        if L == 0 or L == 1:
            L_tmp = L if L == 1 else info_dict['K_history'][-1]
            L_vals.append(L_tmp)

            blocked_loss.append(loss)
            sparse_loss.append(loss)
        elif blocked:
            L_vals.append(L)
            blocked_loss.append(loss)
        else:
            sparse_loss.append(loss)

    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    if title: plt.title(title)
    plt.show()

    # Plot loss vs L
    plt.plot(L_vals[:-1], sparse_loss[:-1], label='One-pass sparse model', marker='o')
    plt.plot(L_vals[:-1], blocked_loss[:-1], label='Two-pass sparse model', marker='o')
    plt.hlines(sparse_loss[-1], L_vals[0], L_vals[-2], label='Dense model')
    plt.xlabel('L')
    plt.ylabel('loss')
    plt.legend()
    if title: plt.title(title)
    plt.show()

# Plot clusters (ignore time steps)
def plot_clusters(experiment_out):
    n_plots = len(experiment_out)
    n_rows = np.ceil(n_plots / 2.0)
    
    plt.figure(figsize=(18, n_rows * 4))
    for i, (model, info_dict, label) in enumerate(experiment_out):
        dataset = info_dict['Data']
        plt.subplot(n_rows, 2, i + 1)
        bnpy.viz.PlotComps.plotCompsFromHModel(model, dataset=dataset)
        plt.title(label)
    plt.tight_layout()
    plt.show()
    