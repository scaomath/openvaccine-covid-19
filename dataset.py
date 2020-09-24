from common import *
from lib.net.layer_np import *

data_dir = '/Users/scao/Documents/Coding/openvaccine-covid-19/data'

target_col = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
error_col  = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_Mg_50C', 'deg_error_pH10', 'deg_error_50C']
#target_col = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

rna_dict    = {x:i for i, x in enumerate('ACGU')} #4
struct_dict = {x:i for i, x in enumerate('().')}  #3
loop_dict   = {x:i for i, x in enumerate('BEHIMSX')}#7

# df_train  = pd.read_json(data_dir+'/train.json', lines=True)
# df_test   = pd.read_json(data_dir+'/test.json', lines=True)
# df_sample = pd.read_csv(data_dir+'/sample_submission.csv')

def filter_by_signal_to_noise(index, df, threshold=1):
    accept = np.array(index)[np.where(df.loc[index].signal_to_noise >= threshold )[0]].tolist()
    reject = list(set(index) - set(accept))
    return accept, reject

def filter_by_sn_filter(index, df):
    accept = np.array(index)[np.where(df.loc[index].SN_filter == 1)[0]].tolist()
    reject = list(set(index) - set(accept))
    return accept, reject


def make_train_split():

    df_train  = pd.read_json(data_dir+'/train.json', lines=True)
    index = np.arange(len(df_train))
    random.Random(123).shuffle(index)
    d = int(0.20*len(index))
    num_split = 5

    train_index = []
    valid_index = []
    for i in range(num_split):
        v_index = index[i*d:(i+1)*d].tolist()
        t_index = list(set(index)-set(v_index))
        train_index.append(t_index)
        valid_index.append(v_index)

    return train_index, valid_index

'''

'GGAAAAGCUCUAAUAACAGGAGACUAGGACUACGUAUUUCUAGGUAACUGGAAUAACCCAUACCAGCAGUUAGAGUUCGCUCUAACAAAAGAAACAACAACAACAAC'
'.....(((((((((((((((((((((((....)))))))))).)))))))))))))..(((...))).(((((((....))))))).....................'
'EEEEESSSSISSIIIIISSSSMSSSHHHHHSSSMMSSSSHHHHHHSSSSMMSSSSIIIIISSISSSSXSSSSSSSHHHHSSSSSSSEEEEEEEEEEEEEEEEEEEEE'


'''



class RNADataset(Dataset):
    def __init__(self, df, augment=None):

        self.rna    = df['sequence'].map(lambda seq: [rna_dict[x] for x in seq])
        self.struct = df['structure'].map(lambda seq: [struct_dict[x] for x in seq])
        self.loop   = df['predicted_loop_type'].map(lambda seq: [loop_dict[x] for x in seq])

        if 1:
            bbp = []
            id = df.id.values
            for i in id:
                probability = np.load(data_dir + '/bpps/%s.npy' % i)
                length = len(probability)

                m = probability.max(-1)
                s = probability.sum(-1)
                cut = np.stack([m,s]).T
                bbp.append(cut)
            self.bbp = bbp

        #---
        if 'reactivity' in df.columns:
            target = np.transpose(
                df[target_col].values.tolist(),
            (0, 2, 1))
            target = np.ascontiguousarray(target)

            error = np.transpose(
                df[error_col].values.tolist(),
            (0, 2, 1))
            error = np.ascontiguousarray(error)
            df['min_value'] = target.reshape(len(target),-1).min(-1)

        else:#dummy
            target = np.zeros((len(df),1,1))
            error  = np.zeros((len(df),1,1))
            df['min_value'] = 0
            df['signal_to_noise'] = 1

        self.df =  df
        self.len = len(self.df)
        self.augment = augment
        self.target = target
        self.error  = error

    def __str__(self):
        string  = ''
        string += '\tlen  = %d\n'%len(self)
        return string

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        r = self.df.loc[index]
        target = self.target[index]
        error = self.error[index]

        rna = np.array(self.rna[index])
        struct = np.array(self.struct[index])
        loop = np.array(self.loop[index])
        bbp  = np.array(self.bbp[index])


        seq = np.concatenate([
            np_onehot(rna,4),
            np_onehot(struct,3),
            np_onehot(loop,7),
            bbp,
        ],1)

        #------
        record = {
            'index' : index,
            'id' : r.id,
            'sequence' : r.sequence,
            'signal_to_noise' : r.signal_to_noise,
            'target': target,
            'error': error,
            'seq' : seq,
        }
        if self.augment is not None: record = self.augment(record)
        return record


def null_collate(batch):
    batch_size = len(batch)
    index = []
    signal_to_noise = []
    target = []
    error  = []
    seq = []

    for r in batch:
        index.append(r['index'])
        signal_to_noise.append(r['signal_to_noise'])
        target.append(r['target'])
        error.append(r['error'])
        seq.append(r['seq'])

    error = np.stack(error)
    error = torch.from_numpy(error).float()
    signal_to_noise = np.stack(signal_to_noise)
    signal_to_noise = torch.from_numpy(signal_to_noise).float()

    target = np.stack(target)
    target = torch.from_numpy(target).float()
    seq = np.stack(seq)
    seq = torch.from_numpy(seq).float()
    return seq, target, error, signal_to_noise, index

##################################################################

def run_check_dataset():

    if 1:
        train_index, valid_index = make_train_split()
        print('t_index %d:'%(len(train_index[0])), train_index[0][:8], 'valid_index %d:'%(len(valid_index[0])), valid_index[0][:8], )
        print('t_index %d:'%(len(train_index[1])), train_index[1][:8], 'valid_index %d:'%(len(valid_index[1])), valid_index[1][:8], )
        print('t_index %d:'%(len(train_index[2])), train_index[2][:8], 'valid_index %d:'%(len(valid_index[2])), valid_index[2][:8], )
        print('')
        #exit(0)

    #---

    df = pd.read_json(data_dir+'/train.json', lines=True)
    df = df[df['signal_to_noise']>1].reset_index(drop=False)
    dataset = RNADataset(
        df,
        augment=None,
    )
    print(dataset)

    if 1:
        start_timer = timer()
        for i in range(len(dataset)):
            if i == 10: break

            r = dataset[i]
            print('i:', i)
            print('\tindex:', r['index'])
            print('\tid:', r['id'])
            print('\tsequence:', r['sequence'][:20], '...', r['sequence'][-20:], len(r['sequence']))
            print('\ttarget:', r['target'][:5].tolist(), '...', r['target'].shape)
            print('\terror:', r['error'][:5].tolist(), '...', r['error'].shape)
            print('\tseq:', r['seq'][:5].astype(np.uint8).tolist(), '...', r['seq'].shape)


    if 1:
        data_loader = DataLoader(
            dataset,
            # sampler = SequentialSampler(dataset),
            batch_size=8,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=null_collate
        )
        print(len(data_loader.sampler))
        print(len(data_loader))
        print(len(data_loader) * data_loader.batch_size)
        print(len(dataset))

        start_timer = timer()
        for t, (seq, target, error, index) in enumerate(data_loader):
            if t == 10: break

            print('[%d]' % t, time_to_str(timer() - start_timer, 'min'))
            print('\t index :', index[:5], '...')
            print('\t error :', error.shape, target.is_contiguous())
            print('\t truth :', target.shape, target.is_contiguous())
            print('\t seq :', seq.shape, seq.is_contiguous())
            print('')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset()