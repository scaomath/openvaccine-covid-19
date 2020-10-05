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
import RNA
# %%
pretrain_dir = None # model dir for resuming training. if None, train from scrach
DEBUG = True
one_fold = False # if True, train model at only first fold. use if you try a new idea quickly.
run_test = False # if True, use small data. you can check whether this code run or not
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

#%% explicit un-enscapuslated routine of get_structure_adj for id #1
i = 0
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
a_neighbor = np.diag(np.ones(seq_length-1),-1) + np.diag(np.ones(seq_length-1),1)
a_strc = np.concatenate([a_strc, a_neighbor[...,None]], axis = 2)
a_strc = np.sum(a_strc, axis = 2, keepdims = True)

# base pairing prob plot vs structure
fig, axes = plt.subplots(2, 2, figsize=(10,10))
axes = axes.flatten()
a_strc = a_strc.squeeze()
axes[0].spy(a_strc)
axes[1].spy(As[1])
axes[2].spy(As[1]*(As[1]>1e-2))
axes[3].spy(As[1]*(a_strc>0))

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
## concat adjacent
As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)
As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)
As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)
# del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri
As.shape, As_pub.shape, As_pri.shape

#%%
## sequence
def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp

def get_inverse_distance_to_loop(sequence, loop_type):
    '''
    compute the graph distance of each base to the near loop
    '''
    prev = float('-inf')
    Dist = []
    for i, x in enumerate(sequence):
        if x == loop_type: 
            prev = i
        Dist.append(i - prev)

    prev = float('inf')
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == loop_type: prev = i
        Dist[i] = min(Dist[i], prev - i)
    Dist = 1/(np.array(Dist)+1)**2
    return Dist*(Dist>0.02)

def get_input(train, As):
    '''
    get node features, which is one hot encoded
    S: paired "Stem"
    M: Multiloop
    I: Internal loop
    B: Bulge
    H: Hairpin loop
    E: dangling End
    X: eXternal loop
    '''
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    
    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    print(vocab, '\n')
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)
    X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)

    ## inverse distance to loops and positional entroy
    vocab = ["M", "I", "B", "H", "E", "X"]
    dist_inv_to_loops = np.zeros((train.shape[0], As.shape[1], len(vocab)))
    positional_entropy = np.zeros((train.shape[0], As.shape[1], 1))
    for i in tqdm(range(len(train))):
        idx = train.index[i]
        for j, s in enumerate(vocab):
            dist_inv_to_loops[i,:,j] = get_inverse_distance_to_loop(train["predicted_loop_type"][idx], s)
        # fc = RNA.fold_compound(train['sequence'][idx])
        # mfe_struct, mfe = fc.mfe()
        # fc.exp_params_rescale(mfe)
        # pp, fp = fc.pf()
        # entropy = fc.positional_entropy()
        # positional_entropy[i,:,0] = np.array(entropy)[1:]
    
    X_node = np.concatenate([dist_inv_to_loops, X_node], axis = 2)
    # X_node = np.concatenate([dist_inv_to_loops,positional_entropy, X_node], axis = 2)

    ## positional entropy

    print(X_node.shape, '\n')
    return X_node

X_node = get_input(train, As)
X_node_pub = get_input(test_pub, As_pub)
X_node_pri = get_input(test_pri, As_pri)


#%% ohe of A G C U base
if DEBUG:
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

# ohe of loop
    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : 
                                                list(map(lambda y : 
                                                        mapping[y], list(x)))))
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    # This is not used
    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
#%% this ohe may not be needed
if DEBUG:
    # obtain the number of types of nodes considering all nodal features (Base type x loop type)
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    print(vocab, '\n')
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)

#%%
if DEBUG:
    for i, row in train.iterrows():
        break

    seq = row.sequence
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    pp, fp = fc.pf()
    entropy = fc.positional_entropy()
    print(entropy)

#%% model
import tensorflow as tf
from tensorflow.keras import layers as L
# import tensorflow_addons as tfa 
from tensorflow.keras import backend as K

def mcrmse(t, p, seq_len_target = seq_len_target):
    ## calculate mcrmse score by using numpy
    t = t[:, :seq_len_target]
    p = p[:, :seq_len_target]
    
    score = np.mean(np.sqrt(np.mean(np.mean((p - t) ** 2, axis = 1), axis = 0)))
    return score

def mcrmse_loss(y_target, y_pred, seq_len_target = seq_len_target):
    ## calculate mcrmse score by using tf
    y_target = y_target[:, :seq_len_target]
    y_pred = y_pred[:, :seq_len_target]
    
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.reduce_mean((y_target - y_pred) ** 2, axis = 1), axis = 0)))
    loss += 2e-1*tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.abs(y_target - y_pred), axis = 1), axis = 0)))
    return loss

def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
#     res = tf.expand_dims(res, axis = 3)
#     res = L.Conv2D(16, 3, 1, padding = "same", activation = "relu")(res)
#     res = L.Conv2D(1, 3, 1, padding = "same", activation = "relu")(res)
#     res = tf.squeeze(res, axis = 3)
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def res(x, unit, kernel = 5, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])

def forward(x, unit, kernel = 5, rate = 0.1):
#     h = L.Dense(unit, None)(x)
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
#         h = tf.keras.activations.swish(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h

def adj_attn(x, adj, unit, n = 2, rate = 0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a) ## aggregate neighborhoods
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base(config):
    ## base model architecture 
    ## node, adj -> middle feature
    
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
    
    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis = 3)([adj, adj_learned])
        
    xs = []
    xs.append(node)
    x1 = forward(node, 128, kernel = 3, rate = 0.1)
    x2 = forward(x1, 64, kernel = 6, rate = 0.1)
    x3 = forward(x2, 32, kernel = 15, rate = 0.1)
    x4 = forward(x3, 16, kernel = 30, rate = 0.1)
    x = L.Concatenate()([x1, x2, x3, x4])
    
    for unit in [128, 64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate = 0.0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel = 30)
        
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
        
    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    return model


def get_ae_model(base, config):
    ## denoising auto encoder part
    ## node, adj -> middle feature -> node
    
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")

    x = base([L.SpatialDropout1D(0.4)(node), adj])
    x = forward(x, 128, rate = 0.3)
    p = L.Dense(X_node.shape[2], "sigmoid")(x)
    
#     loss = - tf.reduce_mean(40 * node * tf.math.log(p + 1e-6) + (1 - node) * tf.math.log(1 - p + 1e-6))
    loss = - tf.reduce_mean(node * tf.math.log(p + 1e-6) + (1 - node) * tf.math.log(1 - p + 1e-6))
    model = tf.keras.Model(inputs = [node, adj], outputs = [loss])
    
    opt = get_optimizer()
    model.compile(optimizer = opt, loss = lambda t, y : y)
    return model


def get_model(base, config):
    ## regression part
    ## node, adj -> middle feature -> prediction of targets
    
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
    
    x = base([node, adj])
    x = forward(x, 128, rate = 0.3)
    x = forward(x, 64, rate = 0.1)
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    
    opt = tf.optimizers.Adam()
    model.compile(optimizer = opt, loss = mcrmse_loss)
    return model



# %%
if __name__ == "__main__":
    print("Just a test file, run in script mode")