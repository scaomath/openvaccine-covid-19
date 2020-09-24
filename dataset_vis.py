#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm

plt.rcParams['figure.figsize'] = (10, 7);
#%%
root_dir = './data/'
train_path = os.path.join(root_dir, 'train.json')
test_path = os.path.join(root_dir, 'test.json')
bpps_dir = os.path.join(root_dir, 'bpps')

train = pd.read_json(train_path, lines=True)
test = pd.read_json(test_path, lines=True)
train.head(3)
# %%
for c in train.columns:
    if train[c].dtype == 'object':
        print(c, ':', train[c].apply(lambda x: len(x))[0])
# %%
def plot_sample(idx = 1):
    samp = train[train.index== idx]
    rna = []
    seq = samp.loc[samp.index[0], 'sequence']
    struct = samp.loc[samp.index[0], 'structure']
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{struct}\n{seq}')[0]
    fvm.plot_rna(bg)
    plt.show()
# %%
plot_sample(2)
# %%
