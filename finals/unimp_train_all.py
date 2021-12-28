import os
import argparse
import datetime as dt
import numpy as np
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import math
import dgl
from dgl.dataloading.neighbor import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from dgl.dataloading.pytorch import NodeDataLoader
import tqdm
from models import Unimp, UnimpGraphConvModel, UnimpGraphAttnModel
from utils import load_dgl_graph, time_diff, get_output, load_dgl_graph_kfolds
from model_utils import early_stopper, thread_wrapped_func, EarlyStopping

from sklearn.metrics import accuracy_score

def load_subtensor(node_feats, mask, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = node_feats[input_nodes].to(device)
    input_labels = labels[input_nodes].to(device)
    batch_masks = mask[input_nodes].squeeze().to(device)
    batch_labels = labels[seeds].to(device)
    
    return batch_inputs, batch_labels, input_labels, batch_masks

epsilon = 1 - math.log(2)
def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)  # comment this line to use logistic loss
    return th.mean(y)


def cleanup():
    dist.destroy_process_group()


def gpu_train(proc_id, n_gpus, GPUS,
              graph_data, gnn_model,
              hidden_dim, n_layers, n_classes, fanouts,
              batch_size=32, num_workers=4, epochs=100, message_queue=None, random=False,
              output_folder='./output', need_val=False):

    device_id = GPUS[proc_id]
    print('Use GPU {} for training ......'.format(device_id))

    # ------------------- 1. Prepare data and split for multiple GPUs ------------------- #
    start_t = dt.datetime.now()
    print('Start graph building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    graph, labels, train_nid, val_nid, test_nid, node_feat, train_mask, val_mask, test_mask = graph_data
    
    train_div, _ = divmod(train_nid.shape[0], n_gpus)
    val_div, _ = divmod(val_nid.shape[0], n_gpus)

    # just use one GPU, give all training index to the one GPU
    if n_gpus == 1:
        train_nid_per_gpu = train_nid
#         val_nid_per_gpu = val_nid[proc_id * val_div: ]
    # in case of multiple GPUs, split training index to different GPUs
    else:
        train_nid_per_gpu = train_nid[proc_id * train_div: (proc_id + 1) * train_div]
#         val_nid_per_gpu = val_nid[proc_id * val_div: (proc_id + 1) * val_div]
    
    val_nid_per_gpu = val_nid
    
    
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 2. Build model for multiple GPUs ------------------------------ #
    start_t = dt.datetime.now()
    print('Start Model building at: {}-{} {}:{}:{}'.format(start_t.month,
                                                           start_t.day,
                                                           start_t.hour,
                                                           start_t.minute,
                                                           start_t.second))

    if n_gpus > 1:
        dist_init_method = 'tcp://{}:{}'.format('127.0.0.1', '23456')
        world_size = n_gpus
        dist.init_process_group(backend='nccl',
                                init_method=dist_init_method,
                                world_size=world_size,
                                rank=proc_id)

    in_feat = node_feat.shape[1]
    if gnn_model == 'graphsage':
        model = Unimp(in_feat, hidden_dim, n_layers, n_classes)
    elif gnn_model == 'graphconv':
        model = UnimpGraphConvModel(in_feat, hidden_dim, n_layers, n_classes,
                               norm='both', activation=F.relu, dropout=0.1)
    elif gnn_model == 'graphattn':
        model = UnimpGraphAttnModel(in_feat, hidden_dim, n_layers, n_classes,
                               heads=([2] * n_layers), activation=F.relu)
    else:
        raise NotImplementedError('So far, only support three algorithms: GraphSage, GraphConv, and GraphAttn')

    model = model.to(device_id)

    if n_gpus > 1:
        model = thnn.parallel.DistributedDataParallel(model,
                                                      device_ids=[device_id],
                                                      output_device=device_id)
    e_t1 = dt.datetime.now()
    h, m, s = time_diff(e_t1, start_t)
    print('Model built used: {:02d}h {:02d}m {:02}s'.format(h, m, s))

    # ------------------- 3. Build loss function and optimizer -------------------------- #
    loss_fn = thnn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    stopper = EarlyStopping(10, os.path.join(output_folder, 'dgl_model-' + args.savename + '.pth'))
    #optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4)
#     earlystoper = early_stopper(patience=20, verbose=False)
    if os.path.exists(os.path.join(output_folder, 'dgl_model-' + args.savename + '.pth')):
        stopper.load_checkpoint(model)
    
    # ------------------- 4. Train model  ----------------------------------------------- #
    print('Plan to train {} epoches \n'.format(epochs))
    sampler = MultiLayerNeighborSampler(fanouts)
    if need_val:
        train_dataloader = NodeDataLoader(graph,
                                          train_nid_per_gpu,
                                          sampler,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=num_workers,
                                          )
        val_dataloader = NodeDataLoader(graph,
                                    val_nid_per_gpu,
                                    sampler,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=num_workers,
                                    )
        
    else:
        train_mask = train_mask + val_mask
        train_dataloader = NodeDataLoader(graph,
                                          np.concatenate((train_nid_per_gpu,val_nid_per_gpu)),
                                          sampler,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=False,
                                          num_workers=num_workers,
                                          )
    for epoch in range(epochs):

        # mini-batch for training
        train_loss_list = []
        # train_acc_list = []
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            # forward
            batch_inputs, batch_labels, input_labels, batch_masks = load_subtensor(node_feat, train_mask, labels, seeds, input_nodes, device_id)
            # 输入的label有适应性的修改，0代表unlabeled, 1 ~ num_classes分别代表不同类别
            # 同时训练需要mask掉除train_set的其他label，避免数据泄露
            # train_mask=1 代表是训练集数据
            batch_masks[:len(batch_labels)] = 0
            input_labels = input_labels * batch_masks + 1
            
            # 加入random扰动
            input_labels = input_labels.int()
            if random:
                rand_y = th.randint(1, n_classes+1, (input_labels.shape[0],))
                rand_y = rand_y.int().to(device_id)
                rand = th.rand((input_labels.shape[0],)) < 0.15
                rand.to(device_id)
                input_labels[rand] = rand_y[rand]
            
            blocks = [block.to(device_id) for block in blocks]
            # metric and loss
            train_batch_logits = model(blocks, batch_inputs, input_labels)
            train_loss = custom_loss_function(train_batch_logits, batch_labels)
#             train_loss = loss_fn(train_batch_logits, batch_labels)
            # backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_list.append(train_loss.cpu().detach().numpy())
            tr_batch_pred = th.sum(th.argmax(train_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

            if step % 10 == 0:
                print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_acc:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(train_loss_list),
                                                                                                tr_batch_pred.detach()))
        # mini-batch for validation
        if epoch>9 and need_val:
            val_loss_list = []
            val_acc_list = []
            correct = total = 0
            model.eval()
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                # forward
                batch_inputs, batch_labels, input_labels, batch_masks = load_subtensor(node_feat, train_mask, labels, seeds, input_nodes, device_id)
                blocks = [block.to(device_id) for block in blocks]
                input_labels = input_labels * batch_masks + 1
                input_labels = input_labels.int()
                # metric and loss
                val_batch_logits = model(blocks, batch_inputs, input_labels)
                val_loss = loss_fn(val_batch_logits, batch_labels)

                val_loss_list.append(val_loss.detach().cpu().numpy())
                val_batch_pred = th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels) / th.tensor(batch_labels.shape[0])

                correct += th.sum(th.argmax(val_batch_logits, dim=1) == batch_labels)
                total += batch_labels.shape[0]
            val_acc = correct / total
            early_stop = stopper.step(val_acc, model)
            if early_stop:
                break
            print('In epoch:{:03d} | val_acc:{:.4f}'.format(epoch, val_acc))
    if need_val:
        del val_dataloader
    del train_dataloader

        
    # -------------------------6. Save models --------------------------------------#
    if need_val:
        stopper.load_checkpoint(model)
    else:
        model_path = os.path.join(output_folder, 'dgl_model-' + args.savename + '.pth')
        th.save(model.state_dict(), self.filename)
     
     # -------------------------7. Testing --------------------------------------#

    if len(fanouts)==2:
        sampler = MultiLayerNeighborSampler([50]*len(fanouts))
        test_dataloader = NodeDataLoader(graph,
                                        test_nid,
                                        sampler,
                                        batch_size=600,
                                        drop_last=False,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        )
    else:
        sampler = MultiLayerNeighborSampler([40,40,10])
        test_dataloader = NodeDataLoader(graph,
                                        test_nid,
                                        sampler,
                                        batch_size=200,
                                        drop_last=False,
                                        shuffle=False,
                                        num_workers=num_workers,
                                        )
    model.eval()
    predicts = []
    ids = []
    logits = []
    if need_val:
        train_mask = train_mask + val_mask
    for step, (input_nodes, seeds, blocks) in enumerate(tqdm.tqdm(test_dataloader)):
        # forward
        batch_inputs, batch_labels, input_labels, batch_masks = load_subtensor(node_feat, train_mask, labels, seeds, input_nodes, device_id)
        blocks = [block.to(device_id) for block in blocks]
        input_labels = input_labels * batch_masks + 1
        input_labels = input_labels.int()
        # metric and loss
        test_batch_logits = model(blocks, batch_inputs, input_labels)
        logits.append(test_batch_logits.detach().cpu().numpy())
        predicts.append(th.argmax(test_batch_logits, dim=1).cpu().numpy())
        ids.append(seeds.cpu().numpy())
    predicts = np.hstack(predicts)
    ids = np.hstack(ids)
    logits = np.vstack(logits)
#     get_output(predicts, ids, args.savename +".csv")
    np.savez("logits_"+ args.savename +".npz", ids=ids, logits=logits)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
    parser.add_argument('--data_path', type=str, help="Path of saved processed data files.")
    parser.add_argument('--gnn_model', type=str, choices=['graphsage', 'graphconv', 'graphattn'],
                        required=True, default='graphsage')
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument("--fanout", type=str, required=True, help="fanout numbers", default='20,20')
    parser.add_argument('--batch_size', type=int, required=True, default=1)
    parser.add_argument('--GPU', nargs='+', type=int, required=True)
    parser.add_argument('--num_workers_per_gpu', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--out_path', type=str, required=True, help="Absolute path for saving model parameters")
    parser.add_argument('--graphpath', type=str, default='graph.bin')
    parser.add_argument('--savename', type=str, required=True)
    parser.add_argument('--random', type=bool, required=False)
    parser.add_argument('--n2v', type=int, default=1)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--fold', type=int, default=-1)
    args = parser.parse_args()

    # parse arguments
    BASE_PATH = args.data_path
    MODEL_CHOICE = args.gnn_model
    HID_DIM = args.hidden_dim
    N_LAYERS = args.n_layers
    FANOUTS = [int(i) for i in args.fanout.split(',')]
    BATCH_SIZE = args.batch_size
    GPUS = args.GPU
    WORKERS = args.num_workers_per_gpu
    EPOCHS = args.epochs
    OUT_PATH = args.out_path

    # output arguments for logging
    print('Data path: {}'.format(BASE_PATH))
    print('Used algorithm: {}'.format(MODEL_CHOICE))
    print('Hidden dimensions: {}'.format(HID_DIM))
    print('number of hidden layers: {}'.format(N_LAYERS))
    print('Fanout list: {}'.format(FANOUTS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('GPU list: {}'.format(GPUS))
    print('Number of workers per GPU: {}'.format(WORKERS))
    print('Max number of epochs: {}'.format(EPOCHS))
    print('Output path: {}'.format(OUT_PATH))

    # Retrieve preprocessed data and add reverse edge and self-loop
    graph, labels, train_nid, val_nid, test_nid, node_feat, train_mask, val_mask, test_mask = load_dgl_graph_kfolds(BASE_PATH, fold=args.fold, k=10, graph_name=args.graphpath ,n2v=args.n2v)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    # call train with CPU, one GPU, or multiple GPUs
    
    n_gpus = len(GPUS)

    
    gpu_train(0, n_gpus, GPUS,
              graph_data=(graph, labels, train_nid, val_nid, test_nid, node_feat, train_mask, val_mask, test_mask),
              gnn_model=MODEL_CHOICE, hidden_dim=HID_DIM, n_layers=N_LAYERS, n_classes=23,
              fanouts=FANOUTS, batch_size=BATCH_SIZE, num_workers=WORKERS, epochs=EPOCHS,
              message_queue=None, output_folder=OUT_PATH, random=args.random,need_val=args.val)
    
