#-*- coding:utf-8 -*-

"""
    Utilities to handel graph data
"""

import os
import dgl
import pickle
import numpy as np
import torch as th
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def load_dgl_graph_kfolds(base_path, fold=-1, k=10, graph_name='graph.bin', dim_reduction=False, emb=True, n2v=1, seed=367):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, graph_name))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    if fold == -1:
        tr_label_idx = label_data['tr_label_idx']
        val_label_idx = label_data['val_label_idx']
    else:
        train_idx = np.concatenate((label_data['tr_label_idx'], label_data['val_label_idx']))   
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (tr, val) in enumerate(folds.split(train_idx, labels[train_idx])):
            tr_label_idx, val_label_idx = train_idx[tr], train_idx[val]
            if i == fold:
                print('    ###      use      fold: {}'.format(fold))
                break
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    
    
    
    
    # random walk特征和邻居特征
#     rw_features = np.load(os.path.join(base_path, 'walk_label_features.npy'))
#     nb_features = np.load(os.path.join(base_path, 'node_info.npy'))
#     features_last = np.hstack((rw_features, nb_features))
#     features_last = nb_features
#     features_last = np.zeros((features.shape[0],1))
    
    if dim_reduction:
        pca = PCA(n_components=196, svd_solver='arpack')
        pca.fit(features)
        features = pca.transform(features)
    if emb:
        if n2v==1:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_1.npy")
        elif n2v==2:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_2.npy")
        elif n2v==3:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/features_n2v_64.npy")
        elif n2v==4:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/features_n2v_64_iso.npy")
        else:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_cite.npy")
        features = np.hstack([features, embs])
    std = StandardScaler()
    features=std.fit_transform(features)
    # 出度入度
    count_features = np.load(os.path.join(base_path, 'count_features.npy'))
    features = np.hstack((features, count_features))
    
    node_feat = th.from_numpy(features).float()
        
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))
    
    # set train, val, test masks
    train_mask = np.zeros((features.shape[0],1))
    train_mask[tr_label_idx, 0] = 1
    val_mask = np.zeros((features.shape[0],1))
    val_mask[val_label_idx, 0] = 1
    test_mask = np.zeros((features.shape[0],1))
    test_mask[test_label_idx, 0] = 1
    train_mask = th.from_numpy(train_mask).float()
    test_mask = th.from_numpy(test_mask).float()
    val_mask = th.from_numpy(val_mask).float()
    
    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat, train_mask, val_mask, test_mask


def load_dgl_graph(base_path, graph_name='graph.bin', dim_reduction=False, emb=True, n2v=1):
    """
    读取预处理的Graph，Feature和Label文件，并构建相应的数据供训练代码使用。

    :param base_path:
    :return:
    """
    graphs, _ = dgl.load_graphs(os.path.join(base_path, graph_name))
    graph = graphs[0]
    print('################ Graph info: ###############')
    print(graph)

    with open(os.path.join(base_path, 'labels.pkl'), 'rb') as f:
        label_data = pickle.load(f)

    labels = th.from_numpy(label_data['label'])
    tr_label_idx = label_data['tr_label_idx']
    val_label_idx = label_data['val_label_idx']
    test_label_idx = label_data['test_label_idx']
    print('################ Label info: ################')
    print('Total labels (including not labeled): {}'.format(labels.shape[0]))
    print('               Training label number: {}'.format(tr_label_idx.shape[0]))
    print('             Validation label number: {}'.format(val_label_idx.shape[0]))
    print('                   Test label number: {}'.format(test_label_idx.shape[0]))

    # get node features
    features = np.load(os.path.join(base_path, 'features.npy'))
    # features标准化
    std = StandardScaler()
    features=std.fit_transform(features)
    
    # 出度入度
    count_features = np.load(os.path.join(base_path, 'count_features.npy'))
    features = np.hstack((features, count_features))
    
    # random walk特征和邻居特征
#     rw_features = np.load(os.path.join(base_path, 'walk_label_features.npy'))
#     nb_features = np.load(os.path.join(base_path, 'node_info.npy'))
#     features_last = np.hstack((rw_features, nb_features))
#     features_last = nb_features
    
    if dim_reduction:
        pca = PCA(n_components=196, svd_solver='arpack')
        pca.fit(features)
        features = pca.transform(features)
    if emb:
        if n2v==1:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_1.npy")
        elif n2v==2:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_2.npy")
        else:
            embs = np.load("/home/chenyu_tian/maxpcontest_model/final_data/n2v_cite.npy")
        features = np.hstack([features, embs])
    node_feat = th.from_numpy(features).float()
    node_feat_concat = th.from_numpy(features_last).float()
    print('################ Feature info: ###############')
    print('Node\'s feature shape:{}'.format(node_feat.shape))
    print('Node\'s feature shape:{}'.format(node_feat_concat.shape))
    
    # set train, val, test masks
    train_mask = np.zeros((features.shape[0],1))
    train_mask[tr_label_idx, 0] = 1
    val_mask = np.zeros((features.shape[0],1))
    val_mask[val_label_idx, 0] = 1
    test_mask = np.zeros((features.shape[0],1))
    test_mask[test_label_idx, 0] = 1
    train_mask = th.from_numpy(train_mask).float()
    test_mask = th.from_numpy(test_mask).float()
    val_mask = th.from_numpy(val_mask).float()
    
    return graph, labels, tr_label_idx, val_label_idx, test_label_idx, node_feat, train_mask, val_mask, test_mask


def time_diff(t_end, t_start):
    """
    计算时间差。t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def get_output(predicts, test_label_idx, path):
    output = {'node_idx':test_label_idx, 'label':predicts}
    output_df = pd.DataFrame(output)
    nodes_df = pd.read_csv("/home/chenyu_tian/maxpcontest_model/data/IDandLabels.csv")
#     sample_submission = pd.read_csv("/home/chenyu_tian/maxpcontest_model/data/sample_submission_for_validation.csv")['id'].tolist()
    output_df = output_df.merge(nodes_df, how='left', left_on='node_idx', right_on='node_idx')
    output_df = output_df[['paper_id','label']]
#     output_df = output_df[output_df['paper_id'].isin(sample_submission)] 
    output_df.columns = [['id', 'label']]

    output_df.to_csv(path, index=False)
    df = pd.read_csv(path)
    df['label'] = df['label'].apply(lambda x: chr(ord('A') + int(x)))
#     output_df['label'] = output_df['label'].apply(lambda x: chr(ord('A') + int(x)))
    df.to_csv(path, index=False)
    
def get_logits_output(logits, path):
    output_df = pd.DataFrame(logits)
    
    output_df.to_csv(path, index=False)