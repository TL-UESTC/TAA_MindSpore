from dis import dis
import tqdm
# import torch
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore
import mindspore.context as context
import mindspore.numpy as mnp
import pandas as pd

from datasets.aliexpress import AliExpressDataset as AliExpressDataset1
from models.aitm import AITMModel
from models.sharedbottom import SharedBottomModel
# from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe_da import MMoEModel
from models.ple import PLEModel
# from models.metaheac import MetaHeacModel
from models.discriminator import Discriminator
from models.cell import DisWithLossCell0,DisWithLossCell1,GenWithLossCell

import warnings

warnings.filterwarnings('ignore')
import time

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.common.dtype as mstype
from mindspore.communication import get_rank, get_group_size
from mindspore.communication import init

# mindspore.dataset.config.set_enable_autotune(True)

class AliExpressDataset():
    def __init__(self, data):
        # print(dataset_path)
        # data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
        self.categorical_data = data[:, :16].astype(np.int)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset1(path)
    else:
        raise ValueError('unknown dataset name: ' + name)
    
def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                 tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    # elif name == 'singletask':
    #     print("Model: SingleTask")
    #     return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
    #                            tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                        tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
                        specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
    # if name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    # elif name == 'metaheac':
    #     print("Model: MetaHeac")
    #     return MetaHeacModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
    #                          tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, critic_num=5,
    #                          dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)
    
class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            mindspore.save_checkpoint(model, self.save_path)
            print('-' * 20, 'Save Model Success', '-' * 20)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(tgt_model, optimizer, src_model, src_optimizer, dis_model, dis_optimizer, train_src_loader, train_tgt_loader,
          criterion, device, task_num,learning_c,learning_p,learning_e,learning_e2,log_interval=100):
    tgt_model.set_train()
    dis_model0, dis_model1 = dis_model
    dis_optimizer0, dis_optimizer1 = dis_optimizer
    dis_model0.set_train()
    dis_model1.set_train()
    total_loss = 0
    src_loader = tqdm.tqdm(train_src_loader, smoothing=0, mininterval=1.0)
    tgt_loader = tqdm.tqdm(train_tgt_loader, smoothing=0, mininterval=1.0)
    total_dis_loss = 0.0
    total_tgt_loss = 0.0
    n_critic_count = 0
    num_dis = 0
    num_tgt = 0

    netG_train = nn.TrainOneStepCell(GenWithLossCell(tgt_model,dis_model0,dis_model1,criterion),optimizer)
    netD0_train = nn.TrainOneStepCell(DisWithLossCell0(dis_model0),dis_optimizer0)
    netD1_train = nn.TrainOneStepCell(DisWithLossCell1(dis_model1),dis_optimizer1)

    netG_train.set_train()
    netD0_train.set_train()
    netD1_train.set_train()
    for p in src_model.trainable_params():
        p.requires_grad = False
    src_model.set_train(False)

    for p in tgt_model.trainable_params():

        p.requires_grad = True

    for i, data_pack in enumerate(zip(src_loader, tgt_loader)):
        src_pack, tgt_pack = data_pack
        s_categorical_fields, s_numerical_fields, s_labels = src_pack
        t_categorical_fields, t_numerical_fields, t_labels = tgt_pack
        # s_categorical_fields, s_numerical_fields, s_labels = s_categorical_fields.to(device), s_numerical_fields.to(
        #     device), s_labels.to(device)
        # t_categorical_fields, t_numerical_fields, t_labels = t_categorical_fields.to(device), t_numerical_fields.to(
        #     device), t_labels.to(device)
        if s_categorical_fields.shape[0] != t_categorical_fields.shape[0]:
            break

        for p in dis_model0.trainable_params():
            p.requires_grad = True

        # dis_model0.zero_grad()

        for p in dis_model1.trainable_params():
            p.requires_grad = True

        _, src_feat1 = tgt_model(s_categorical_fields, s_numerical_fields)
        if not isinstance(src_feat1, list):
            src_feat1 = [src_feat1]*task_num

        src_feat = src_feat1

        _, tgt_feat = tgt_model(t_categorical_fields, t_numerical_fields)
        if not isinstance(tgt_feat, list):
            tgt_feat = [tgt_feat]*task_num

        D0_cost = netD0_train(src_feat,tgt_feat)
        D1_cost = netD0_train(src_feat,tgt_feat)
        
        num_dis += 1
        n_critic_count += 1

        if n_critic_count >= 5:
            for p in dis_model0.trainable_params():
                p.requires_grad = False

            for p in dis_model1.trainable_params():
                p.requires_grad = False

            loss = netG_train(t_categorical_fields,t_numerical_fields,t_labels,task_num,learning_e,learning_e2,learning_c,learning_p)

            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tgt_loader.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
            n_critic_count = 0

def metatrain(model, optimizer, data_loader, device, log_interval=100):
    model.set_train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        # categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
        #     device), labels.to(device)
        batch_size = int(categorical_fields.size(0) / 2)
        list_sup_categorical.append(categorical_fields[:batch_size])
        list_qry_categorical.append(categorical_fields[batch_size:])
        list_sup_numerical.append(numerical_fields[:batch_size])
        list_qry_numerical.append(numerical_fields[batch_size:])
        list_sup_y.append(labels[:batch_size])
        list_qry_y.append(labels[batch_size:])

        if (i + 1) % 2 == 0:
            loss = model.global_update(list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical,
                                       list_qry_numerical, list_qry_y)
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item()
            list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, task_num, device):
    model.set_train(False)
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    # with torch.no_grad():
    for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        # categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
        #     device), labels.to(device)
        y, _ = model(categorical_fields, numerical_fields)
        for i in range(task_num):
            labels_dict[i].extend(labels[:, i].asnumpy().tolist())
            predicts_dict[i].extend(y[i].asnumpy().tolist())
            # predicts_dict[i] = mindspore.Tensor(predict_values)
            # roc_auc_score(labels_dict[i], predict_values)
            loss_dict[i].extend(
                ops.binary_cross_entropy(y[i],ops.Cast()(labels[:, i],mindspore.float32), reduction='none').asnumpy().tolist())

    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i],predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results

import random

def main(tgt_dataset_name,
         src_dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir,
         read_dir,
         chunksize,
         learning_rateda,
         learning_c,
         learning_p,
         learning_e,
         learning_e2):
    # device = torch.device(device)
    field_dims = {
        'AliExpress_NL': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_ES': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_FR': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_US': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_RU': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
    }

    # mindspore.set_auto_parallel_context(parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL, gradients_mean=True)

    # {time.strftime("%m%d_%H%M%S")}
    # test_dataset = get_dataset(tgt_dataset_name, os.path.join(dataset_path, tgt_dataset_name) + '/test.csv')
    # test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    save_path = f'{save_dir}/{src_dataset_name}_{tgt_dataset_name.split("_")[-1]}_{model_name}_{time.strftime("%m%d_%H%M%S")}.ckpt'
    print(save_path)
    # src_read_path = f'{read_dir}/{src_dataset_name}_{model_name}.ckpt'
    # tgt_read_path = src_read_path#f'{read_dir}/{tgt_dataset_name}_{model_name}.ckpt'

    
    dis_model0 = Discriminator(input_dims=256, hidden_dims=64)
    dis_model1 = Discriminator(input_dims=256, hidden_dims=64)
    

    # dis_optimizer0 = torch.optim.Adam(params=dis_model0.parameters(), lr=learning_rateda, weight_decay=weight_decay)
    dis_optimizer0 = nn.optim.Adam(params = dis_model0.trainable_params(), learning_rate=learning_rateda, weight_decay = weight_decay)
    ##dual
    # dis_model1 = Discriminator(input_dims=256 * task_num, hidden_dims=64).to(device)
    # dis_optimizer1 = torch.optim.Adam(params=dis_model1.parameters(), lr=learning_rateda, weight_decay=weight_decay)
    dis_optimizer1 = nn.optim.Adam(params = dis_model1.trainable_params(), learning_rate=learning_rateda, weight_decay = weight_decay)

    ##
    dis_model = [dis_model0, dis_model1]
    dis_optimizer = [dis_optimizer0, dis_optimizer1]

    criterion = nn.BCELoss()
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    tgt_field_dims = field_dims[tgt_dataset_name]
    src_field_dims = field_dims[src_dataset_name]
    tgt_numerical_num, src_numerical_num = 63, 63
    src_model = get_model(model_name, src_field_dims, src_numerical_num, task_num, expert_num, embed_dim)
    tgt_model = get_model(model_name, tgt_field_dims, tgt_numerical_num, task_num, expert_num, embed_dim)
    # tgt_optimizer = torch.optim.Adam(params=tgt_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tgt_optimizer = nn.optim.Adam(params = tgt_model.trainable_params(), learning_rate=learning_rate, weight_decay = weight_decay)
    # src_optimizer = torch.optim.Adam(params=src_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    src_optimizer = nn.optim.Adam(params = src_model.trainable_params(), learning_rate=learning_rate, weight_decay = weight_decay)
    # src_model.load_state_dict(torch.load(src_read_path))
    mindspore.load_param_into_net(src_model,(mindspore.load_checkpoint("/data2/liuguodong/DA/chkptda/AliExpress_RU_FR_sharedbottom_1107_125714.ckpt")))
    # tgt_model.load_state_dict(torch.load(tgt_read_path))
    mindspore.load_param_into_net(tgt_model,(mindspore.load_checkpoint("/data2/liuguodong/DA/chkptda/AliExpress_RU_FR_sharedbottom_1107_125714.ckpt")))


    # rank_id = get_rank()
    # rank_size = get_group_size()
    ###vanilla cross test

    auc_results, loss_results = [], []
    test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv',
                             chunksize=batch_size * chunksize)
    for i, test_data in enumerate(test_datas):
        test_data = test_data.to_numpy()[:, 1:]
        test_data = AliExpressDataset(test_data)
        # test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
        test_data_loader = mindspore.dataset.GeneratorDataset(test_data,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False, num_parallel_workers = 1)
        test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)       
        auc, loss = test(tgt_model, test_data_loader, task_num, device)
        auc_results.append(auc)
        loss_results.append(loss)

    auc_results, loss_results = [], []
    test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv',
                             chunksize=batch_size * chunksize)
    for i, test_data in enumerate(test_datas):
        test_data = test_data.to_numpy()[:, 1:]
        test_data = AliExpressDataset(test_data)
        # test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
        test_data_loader = mindspore.dataset.GeneratorDataset(test_data,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False, num_parallel_workers = 1)
        test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)       
        auc, loss = test(tgt_model, test_data_loader, task_num, device)
        auc_results.append(auc)
        loss_results.append(loss)

    auc_results, loss_results = np.array(auc_results), np.array(loss_results)
    aus_ans, loss_ans = [], []
    for k in range(task_num):
        aus_ans.append(np.mean(auc_results[:, k]))
        loss_ans.append(np.mean(loss_results[:, k]))
    
    print('cross-test: auc:', aus_ans)
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}'.format(i, aus_ans[i], loss_ans[i]))

    print("="*20)

    #######
    total_time = 0
    total_epoch =0

    for epoch_i in range(epoch):
        start = time.time()
        tgt_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/split_train.csv',
                                chunksize=batch_size * chunksize)
        # tgt_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/train.csv',
        #                         chunksize=batch_size * chunksize)                                
        src_datas = pd.read_csv(os.path.join(dataset_path, src_dataset_name) + '/train.csv',
                                chunksize=batch_size * chunksize, skiprows=lambda i: i > 0 and random.random() > 0.5)
        test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv',
                                 chunksize=batch_size * chunksize)
        for i, pack in enumerate(zip(tgt_datas, src_datas)):
            tgt_trainset, src_trainset = pack[0].to_numpy()[:, 1:], pack[1].to_numpy()[:, 1:]
            tgt_trainset = AliExpressDataset(tgt_trainset)
            src_trainset = AliExpressDataset(src_trainset)
            # train_src_loader = DataLoader(src_trainset, batch_size=batch_size, num_workers=8, shuffle=True)
            # train_src_loader = mindspore.dataset.GeneratorDataset(src_trainset,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = True,num_parallel_workers = 4,num_shards=rank_size, shard_id=rank_id)
            train_src_loader = mindspore.dataset.GeneratorDataset(src_trainset,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = True,num_parallel_workers = 1)
            train_src_loader = train_src_loader.batch(batch_size, drop_remainder=False)
            # train_tgt_loader = DataLoader(tgt_trainset, batch_size=batch_size, num_workers=8, shuffle=True)
            # train_tgt_loader = mindspore.dataset.GeneratorDataset(tgt_trainset,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = True,num_parallel_workers = 4,num_shards=rank_size, shard_id=rank_id)
            train_tgt_loader = mindspore.dataset.GeneratorDataset(tgt_trainset,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = True,num_parallel_workers = 1)
            train_tgt_loader = train_tgt_loader.batch(batch_size, drop_remainder=False)

            if model_name == 'metaheac':
                metatrain(tgt_model, tgt_optimizer, train_tgt_loader, device)
            else:
                train(tgt_model, tgt_optimizer, src_model, src_optimizer, dis_model, dis_optimizer, train_src_loader,
                      train_tgt_loader, criterion, device,task_num,learning_c,learning_p,learning_e,learning_e2)

        end = time.time()
        total = start-end
        total_time += total
        # epoch evaluate
        auc_results, loss_results = [], []
        for i, test_data in enumerate(test_datas):
            test_data = test_data.to_numpy()[:, 1:]
            test_data = AliExpressDataset(test_data)
            # test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
            # test_data_loader = mindspore.dataset.GeneratorDataset(test_data, column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False, num_parallel_workers = 4,num_shards=rank_size, shard_id=rank_id)
            test_data_loader = mindspore.dataset.GeneratorDataset(test_data, column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False, num_parallel_workers = 1)
            test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)
            auc, loss = test(tgt_model, test_data_loader, task_num, device)
            auc_results.append(auc)
            loss_results.append(loss)
        auc_results, loss_results = np.array(auc_results), np.array(loss_results)
        aus_ans, loss_ans = [], []
        for k in range(task_num):
            aus_ans.append(np.mean(auc_results[:, k]))
            loss_ans.append(np.mean(loss_results[:, k]))

        print('epoch:', epoch_i, 'test: auc:', aus_ans)
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, aus_ans[i], loss_ans[i]))

        total_epoch = epoch_i+1
        if not early_stopper.is_continuable(tgt_model, np.array(aus_ans).mean()):
            print(f'test: best auc: {early_stopper.best_accuracy}')
            break

    
    total_time /=60
    print("Total time:")
    print(total_epoch,"epoch ",round(total_time,4),"min")

    # tgt_model.load_state_dict(torch.load(save_path))
    mindspore.load_param_into_net(tgt_model,(mindspore.load_checkpoint(save_path)))
    auc_results, loss_results = [], []
    test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv', chunksize=batch_size * chunksize)
    for i, test_data in enumerate(test_datas):
        test_data = test_data.to_numpy()[:, 1:]
        test_data = AliExpressDataset(test_data)
        # test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
        # test_data_loader = mindspore.dataset.GeneratorDataset(test_data,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False,num_parallel_workers = 4,num_shards=rank_size, shard_id=rank_id)
        test_data_loader = mindspore.dataset.GeneratorDataset(test_data,column_names = ["categorical_data", "numerical_data", "labels"],shuffle = False,num_parallel_workers = 1)
        test_data_loader = test_data_loader.batch(batch_size, drop_remainder=False)
        auc, loss = test(tgt_model, test_data_loader, task_num, device)
        auc_results.append(auc)
        loss_results.append(loss)

    auc_results, loss_results = np.array(auc_results), np.array(loss_results)
    aus_ans, loss_ans = [], []
    for k in range(task_num):
        aus_ans.append(np.mean(auc_results[:, k]))
        loss_ans.append(np.mean(loss_results[:, k]))

    f = open('{}_{}.txt'.format(model_name, tgt_dataset_name), 'a', encoding='utf-8')
    f.write('main_da.py | {} | Time: {}\n'.format(model_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    f.write('Save Path: {}\n'.format(save_path))
    f.write('Source Domain: {}->{}\n'.format(src_dataset_name, tgt_dataset_name))
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}'.format(i, aus_ans[i], loss_ans[i]))
        f.write('task {}, AUC {}, Log-loss {}\n'.format(i, aus_ans[i], loss_ans[i]))

    print('\n')
    f.write('\n')
    f.close()
    #     auc, loss = test(tgt_model, test_data_loader, task_num, device)
    #     print('epoch:', epoch_i, 'test: auc:', auc)
    #     for i in range(task_num):
    #         print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
    #     if not early_stopper.is_continuable(tgt_model, np.array(auc).mean()):
    #         print(f'test: best auc: {early_stopper.best_accuracy}')
    #         break
    #
    #
    # tgt_model.load_state_dict(torch.load(save_path))
    # auc, loss = test(tgt_model, test_data_loader, task_num, device)
    # f = open('{}_{}.txt'.format(model_name, tgt_dataset_name), 'a', encoding = 'utf-8')
    # f.write('learning rate: {}\n'.format(learning_rate))
    # for i in range(task_num):
    #     print('task {}, AUC {}, Log-loss {}'.format(i, auc[i], loss[i]))
    #     f.write('task {}, AUC {}, Log-loss {}\n'.format(i, auc[i], loss[i]))
    # print('\n')
    # f.write('\n')
    # f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dataset_name', default='AliExpress_FR',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'AliExpress_RU'])
    parser.add_argument('--src_dataset_name', default='AliExpress_RU',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'AliExpress_RU'])
    parser.add_argument('--dataset_path', default='/data2/liuguodong/DA/AliExpress/')
    parser.add_argument('--model_name', default='sharedbottom',
                        choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    # parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device_target',default="GPU",choices=['GPU', 'CPU','Ascend'])
    parser.add_argument('--device_id', type=int, default=2)
    parser.add_argument('--save_dir', default='/data2/liuguodong/DA/chkptda')
    parser.add_argument('--read_dir', default='/data2/liuguodong/DA/chkpt_src')
    parser.add_argument('--chunksize', type=int, default=1024)
    parser.add_argument('--learning_rateda', type=float, default=0.1)
    parser.add_argument('--learning_c', type=float, default=1)
    parser.add_argument('--learning_p', type=float, default=1)
    parser.add_argument('--learning_e', type=float, default=1)
    parser.add_argument('--learning_e2', type=float, default=1)
    args = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
        device_id = args.device_id)
    # init("nccl")
    main(args.tgt_dataset_name,
         args.src_dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device_target,
         args.save_dir,
         args.read_dir,
         args.chunksize,
         args.learning_rateda,
         args.learning_c,
         args.learning_p,
         args.learning_e,
         args.learning_e2)