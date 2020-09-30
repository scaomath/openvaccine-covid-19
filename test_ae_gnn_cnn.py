#%%
import shutil
import glob
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import json
from tqdm import tqdm
%matplotlib inline

# %%
pretrain_dir = None # model dir for resuming training. if None, train from scrach

one_fold = False # if True, train model at only first fold. use if you try a new idea quickly.
run_test = True # if True, use small data. you can check whether this code run or not
denoise = True # if True, use train data whose signal_to_noise > 1

ae_epochs = 20 # epoch of training of denoising auto encoder
ae_epochs_each = 5 # epoch of training of denoising auto encoder each time. 
                   # I use train data (seqlen = 107) and private test data (seqlen = 130) for auto encoder training.
                   # I dont know how to easily fit keras model to use both of different shape data simultaneously, 
                   # so I call fit function several times. 
ae_batch_size = 32

epochs_list = [30, 10, 3, 3, 5, 5]
batch_size_list = [8, 16, 32, 64, 128, 256] 
if pretrain_dir is not None:
    for d in glob.glob(pretrain_dir + "*"):
        shutil.copy(d, ".")
# %%
data_dir = './data/'
train = pd.read_json(data_dir+"train.json",lines=True)
if denoise:
    train = train[train.signal_to_noise > 1].reset_index(drop = True)
test  = pd.read_json(data_dir+"test.json",lines=True)
test_pub = test[test["seq_length"] == 107]
test_pri = test[test["seq_length"] == 130]
sub = pd.read_csv(data_dir+"/sample_submission.csv")

if run_test: ## to test 
    train = train[:30]
    test_pub = test_pub[:30]
    test_pri = test_pri[:30]

As = []
for id in tqdm(train["id"]):
    a = np.load(f"{data_dir}/bpps/{id}.npy")
    As.append(a)
As = np.array(As)
As_pub = []
for id in tqdm(test_pub["id"]):
    a = np.load(f"{data_dir}/bpps/{id}.npy")
    As_pub.append(a)
As_pub = np.array(As_pub)
As_pri = []
for id in tqdm(test_pri["id"]):
    a = np.load(f"{data_dir}/bpps/{id}.npy")
    As_pri.append(a)
As_pri = np.array(As_pri)
# %%

targets = list(sub.columns[1:])
print(targets)

y_train = []
seq_len = train["seq_length"].iloc[0]
seq_len_target = train["seq_scored"].iloc[0]
ignore = -10000
ignore_length = seq_len - seq_len_target
for target in targets:
    y = np.vstack(train[target])
    dummy = np.zeros([y.shape[0], ignore_length]) + ignore
    y = np.hstack([y, dummy])
    y_train.append(y)
y = np.stack(y_train, axis = 2)
# %%
def get_structure_adj(train):
    ## get adjacent matrix from structure sequence
    
    ## here I calculate adjacent matrix of each base pair, 
    ## but eventually ignore difference of base pair and integrate into one matrix
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
#                 a_structure[start, i] = 1
#                 a_structure[i, start] = 1
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss
Ss = get_structure_adj(train)
Ss_pub = get_structure_adj(test_pub)
Ss_pri = get_structure_adj(test_pri)


#%% explicit un-enscapuslated routine of get_structure_adj
i=1
seq_length = train["seq_length"].iloc[i]
structure = train["structure"].iloc[i]
sequence = train["sequence"].iloc[i]

cue = []
a_structures = {
    ("A", "U") : np.zeros([seq_length, seq_length]),
    ("C", "G") : np.zeros([seq_length, seq_length]),
    ("U", "G") : np.zeros([seq_length, seq_length]),
    ("U", "A") : np.zeros([seq_length, seq_length]),
    ("G", "C") : np.zeros([seq_length, seq_length]),
    ("G", "U") : np.zeros([seq_length, seq_length]), 
}
a_structure = np.zeros([seq_length, seq_length])
for i in tqdm(range(seq_length)):
    if structure[i] == "(":
        cue.append(i)
    elif structure[i] == ")":
        start = cue.pop()
        a_structures[(sequence[start], sequence[i])][start, i] = 1
        a_structures[(sequence[i], sequence[start])][i, start] = 1
a_strc = np.stack([a for a in a_structures.values()], axis = 2)
a_strc = np.sum(a_strc, axis = 2, keepdims = True)

# %% visualize the neighboring struct in train

def plot_sparse_pattern(Struct_mat, data, sample_size=10):
    n_row = sample_size//5
    fig, axes = plt.subplots(n_row,5, figsize=(n_row*5,5))
    axes = axes.flatten()
    idx = np.random.randint(0,Struct_mat.shape[0],size=n_row*5)
    for i, _i  in enumerate(idx):
        axes[i].spy(sparse.csr_matrix(Struct_mat[_i,:,:,0]))
        axes[i].set_title(data.iloc[_i].id, color= 'black', fontsize=10)
    plt.show()


plot_sparse_pattern(Ss_pri, test_pri)
# %%
def get_distance_matrix(As):
    ## adjacent matrix based on distance on the sequence
    ## D[i, j] = 1 / (abs(i - j) + 1) ** pow, pow = 1, 2, 4
    
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2]: 
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds

Ds = get_distance_matrix(As)
Ds_pub = get_distance_matrix(As_pub)
Ds_pri = get_distance_matrix(As_pri)

# %%
## concat adjecent
As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)
As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)
As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)
# del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri
As.shape, As_pub.shape, As_pri.shape
# %%
if __name__ == "__main__":
    print("Just a test file, run in script mode")