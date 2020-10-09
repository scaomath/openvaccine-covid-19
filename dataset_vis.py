#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm

# plt.rcParams['figure.figsize'] = (8, 7);
# plt.rcParams['axes.facecolor'] = 'gray'
#%%
root_dir = './data/'
train_path = os.path.join(root_dir, 'train.json')
test_path = os.path.join(root_dir, 'test.json')
bpps_dir = os.path.join(root_dir, 'bpps')

train = pd.read_json(train_path, lines=True)
# train = train[train.signal_to_noise > 1].reset_index(drop = True)
test = pd.read_json(test_path, lines=True)
train.head(3)
# %%
for c in train.columns:
    if train[c].dtype == 'object':
        print(c, ':', train[c].apply(lambda x: len(x))[0])
# %%
def plot_sample(idx = 1, sequence=None, df=train):
    fig = plt.figure(figsize=(15,15))
    fig.patch.set_facecolor((160/255, 177/255, 198/255))
    if sequence is not None:
        samp = df[df.id == sequence]
    else:
        samp = df[df.index == idx]
    rna = []
    seq = samp.loc[samp.index[0], 'sequence']
    struct = samp.loc[samp.index[0], 'structure']
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]
    fvm.plot_rna(bg, 
                text_kwargs={"fontweight":"bold",
                            }, 
                lighten=0.8,
                backbone_kwargs={"linewidth":3})
    plt.show()
# %%
plot_sample(idx=2)
# %%
plot_sample(sequence='id_7704f616f', df = test)

# %%
plot_sample(sequence='id_542027cb0', df = train)
# %%
