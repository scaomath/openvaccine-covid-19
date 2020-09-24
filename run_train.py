from common import *
from lib.net.radam import *
from lib.net.lookahead import *
from lib.net.layer_np import *

from dataset import *
from model import *


def np_mc_rmse(predict, target):
    predict = predict.reshape(-1,5)
    target = target.reshape(-1,5)

    error = (predict-target)**2
    error = error.mean(0)
    error = error**0.5

    return error



def do_valid(net, valid_loader, is_mixed_precision):

    valid_predict = []
    valid_target = []
    valid_index = []
    start_timer = timer()
    with torch.no_grad():
        net.eval()

        valid_num = 0
        for t, (seq, target, error, signal_to_noise, index) in enumerate(valid_loader):
            # seq = seq.cuda()

            if is_mixed_precision:
                with amp.autocast():
                    pass
            else:
                predict = net(seq)  # data_parallel(net, (sequence, aux))

            valid_predict.append(predict.data.cpu().numpy())
            valid_target.append(target.data.cpu().numpy())
            valid_index.extend(index)
            valid_num += len(index)

            #---
            print('\r %8d / %d  %s'%(valid_num, len(valid_loader.sampler),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
            #if valid_num==200*4: break

        assert(valid_num == len(valid_loader.sampler))
        #print('')

    #------
    target  = np.concatenate(valid_target)
    predict = np.concatenate(valid_predict)
    num, length, num_target = target.shape
    predict = predict[:,:length]

    index, _ = filter_by_sn_filter(np.arange(num).tolist(), valid_loader.dataset.df)
    target  = target[index]
    predict = predict[index]
    error = np_mc_rmse(predict, target)

    return [error[:5].mean(), error[:3].mean(), *error]





def run_train():


    for fold in [1,2,3,4]:#,1,2,3,4  #0, 1,2,3,4
        out_dir = '/Users/scao/Documents/Coding/openvaccine-covid-19/result/sn1-fold-%d'%fold
        initial_checkpoint = None
             #out_dir + '/checkpoint/00007200_model.pth' #

        is_mixed_precision = False #False #True #
        start_lr   = 0.00125#1
        batch_size = 32 #32
        iter_accum = 1 #16

        snr_threshold = 1

        ## setup  ----------------------------------------
        for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
        #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

        log = Logger()
        log.open(out_dir+'/log.train.txt',mode='a')
        log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        log.write('\t%s\n' % COMMON_STRING)
        log.write('\t__file__ = %s\n' % __file__)
        log.write('\tout_dir  = %s\n' % out_dir)
        log.write('\n')


        ## dataset ----------------------------------------
        df = pd.read_json(data_dir+'/train.json', lines=True)

        train_index, valid_index = make_train_split()
        train_index, valid_index = train_index[fold], valid_index[fold]
        train_index, _ = filter_by_signal_to_noise(train_index, df, snr_threshold)

        print('train_index:', train_index[:10], 'valid_index:', valid_index[:10],)
        if 1: #check overlap
            print(set(train_index).intersection(set(valid_index)))

        train_dataset = RNADataset(df.loc[train_index].reset_index(drop=True))
        train_loader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            # sampler = TrainSampler(train_dataset,batch_size,0.2),
            batch_size  = batch_size,
            drop_last   = True,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
        )
        valid_dataset = RNADataset(df.loc[valid_index].reset_index(drop=True))
        valid_loader = DataLoader(
            valid_dataset,
            sampler=SequentialSampler(valid_dataset),
            batch_size  = 128,
            drop_last   = False,
            num_workers = 0,
            pin_memory  = True,
            collate_fn  = null_collate
        )


        log.write('fold = %d\n'%(fold))
        log.write('snr_threshold = %f\n'%(snr_threshold))
        log.write('train_dataset : \n%s\n'%(train_dataset))
        log.write('valid_dataset : \n%s\n'%(valid_dataset))
        log.write('\n')


        ## net ----------------------------------------
        log.write('** net setting **\n')

        if is_mixed_precision:
            scaler = amp.GradScaler() #not used
            # net = AmpNet().cuda()
            net = AmpNet()
        else:
            net = Net()
            #print(net)
            # net = net.cuda()


        #-----
        if initial_checkpoint is not None:
            f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
            start_iteration = f['iteration']#75000 #
            start_epoch = f['epoch']#10.95 #
            state_dict = f['state_dict']#f #

            # for k in list(state_dict.keys()):
            #       if any(s in k for s in ['roi']): state_dict.pop(k, None)

            net.load_state_dict(state_dict,strict=False)  #True
        else:
            start_iteration = 0
            start_epoch = 0
            #net.load_pretrain(is_print=False)

        log.write('net=%s\n'%(type(net)))
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        log.write('\n')

        ## optimiser ----------------------------------
        if 0:  ##freeze
            for p in net.stem.parameters(): p.requires_grad = False
            for p in net.block1.parameters(): p.requires_grad = False
            pass

        # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),  lr=start_lr, momentum=0.9, weight_decay=1e-4)
        # optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),  lr=start_lr)
        optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr), alpha=0.5, k=5)

        num_iteration = int(10.0 * 1000)
        iter_log   = 100
        iter_valid = 100
        iter_save  = list(range(0, num_iteration, 200))  # 1*1000

        log.write('optimizer\n  %s\n' % (optimizer))
        # log.write('schduler\n  %s\n'%(schduler))
        log.write('\n')

        ## start training here! ##############################################
        log.write('** start training here! **\n')
        log.write('   is_mixed_precision = %s \n' % str(is_mixed_precision))
        log.write('   batch_size * iter_accum = %d * %d\n' % (batch_size, iter_accum))
        log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
        log.write('                      |------------------------- VALID--------------------------|---- TRAIN/BATCH --------------\n')
        log.write('                      |    mc_rmse    |      rmse (Mg)        |      rmse       | loss0  loss1  |               \n')
        log.write('rate     iter   epoch | col-5  col-3  |  react  ph10    50c   |  ph10     50c   | col-5         | time          \n')
        log.write('----------------------------------------------------------------------------------------------------------------\n')
                 # 0.00250   0.06*  14.6 | 0.251  0.254  |  0.232  0.290  0.241  |  0.272   0.220  | 0.236  0.000  |  0 hr 00 min

        def message(mode='print'):
            if mode == ('print'):
                asterisk = ' '
                loss = batch_loss
            if mode == ('log'):
                asterisk = '*' if iteration in iter_save else ' '
                loss = train_loss

            text = \
                '%4.5f  %5.4d %s %5.1f | ' % (rate, iteration, asterisk, epoch,) + \
                '%4.3f  %4.3f  |  %4.3f  %4.3f  %4.3f  |  %4.3f   %4.3f  | ' % (*valid_loss,) + \
                '%4.3f  %4.3f  | ' % (*loss,) + \
                '%s' % (time_to_str(timer() - start_timer, 'min'))

            return text


        # ----
        valid_loss = np.zeros(7, np.float32)
        train_loss = np.zeros(2, np.float32)
        batch_loss = np.zeros_like(train_loss)
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = 0
        loss0 = torch.FloatTensor([0]).sum()
        loss1 = torch.FloatTensor([0]).sum()

        start_timer = timer()
        iteration = start_iteration
        epoch = start_epoch
        rate = 0
        while iteration < num_iteration:

            optimizer.zero_grad()
            for t,  (seq, target, error, signal_to_noise, index)  in enumerate(train_loader):
                #print('ok')

                if (iteration % iter_valid==0):###
                    valid_loss = do_valid(net, valid_loader, is_mixed_precision) #
                    pass

                if (iteration % iter_log==0):
                    print('\r',end='',flush=True)
                    log.write(message(mode='log')+'\n')
                    pass


                if iteration in iter_save:
                    if iteration != start_iteration:
                        torch.save({
                            'state_dict': net.state_dict(),
                            'iteration' : iteration,
                            'epoch' : epoch,
                        #}, out_dir +'/checkpoint/model.pth')
                        }, out_dir +'/checkpoint/%08d_model.pth'%(iteration))
                        pass

                # learning rate schduler -------------
                #adjust_learning_rate(optimizer, schduler(iteration))
                rate = get_learning_rate(optimizer)


                # one iteration update  -------------
                # target = target.cuda()
                # error  = error.cuda()
                # signal_to_noise  = signal_to_noise.cuda()
                # seq = seq.cuda()


                net.train()
                # ----
                if is_mixed_precision:
                    assert(False) #not used
                    with amp.autocast():
                        predict = data_parallel(seq) #net(input)#

                    #-----
                    loss0 = mse_loss(predict,target)
                    scaler.scale((loss0) / iter_accum).backward()
                    if iteration % iter_accum == 0:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                else:
                    #assert(False)
                    predict  = net(seq) #data_parallel(net, (sequence, aux))

                    #loss0 = mse_loss(predict,target)
                    #loss0 = mcrmse_loss(predict,target)
                    loss0 = snr_mcrmse_loss(predict,target, error, signal_to_noise)

                    ((loss0+loss1)/ iter_accum).backward()
                    if iteration % iter_accum == 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                        optimizer.step()
                        optimizer.zero_grad()

                # print statistics  --------
                epoch += 1 / len(train_loader)
                iteration += 1

                batch_loss = np.array([ loss0.item(), loss1.item() ])
                sum_train_loss += batch_loss
                sum_train += 1
                if iteration%100 == 0:
                    train_loss = sum_train_loss/(sum_train+1e-12)
                    sum_train_loss[...] = 0
                    sum_train = 0

                print('\r',end='',flush=True)
                print(message(mode='print'), end='',flush=True)
                #print(iteration, loss.item())



#####################################################################
if __name__ == '__main__':
    run_train()
