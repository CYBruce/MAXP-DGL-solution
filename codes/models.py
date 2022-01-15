#-*- coding:utf-8 -*-

# Author:james Zhang

"""
    Three common GNN models.
"""
import torch as th
import torch.nn as thnn
import torch.nn.functional as F
import dgl.nn as dglnn

# UNIMP-graphsage model
class Unimp(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0.1):
        super(Unimp, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()
        self.filter_convs = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        emb_size = 64
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size, max_norm=True)
        
        # build multiple layers
        self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats+emb_size,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean'))
                                          # aggregator_type = 'pool'))
        for l in range(1, (self.n_layers)):
            self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean'))
                                              # aggregator_type='pool'))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)
        
    def forward(self, blocks, features, y_hat):
        layers=0
        label_embedding = self.label_embedding(y_hat)
        features = th.cat((features, label_embedding), dim=1)
        h = features
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            if l == 0:
                h = gnn(block, h)
            else:
                out = h
                h = gnn(block, h)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out[:block.number_of_dst_nodes()] + gate * h
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            layers+=1
            
        h = self.bn(h)
        out = h
        h = self.MLP1(h)
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h

class UnimpGraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0
                 ):
        super(UnimpGraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        emb_size = 64
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size)
        self.mappings = thnn.ModuleList()
        # build multiple layers
        
        self.layers.append(dglnn.GATConv(in_feats=self.in_feats+emb_size,
                                         out_feats=self.hidden_dim,
                                         num_heads=self.heads[0],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=self.activation,
                                         allow_zero_in_degree=True))
        self.mappings.append(
                            thnn.Sequential(
                            thnn.Linear(self.in_feats+self.hidden_dim+emb_size,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
        
        for l in range(1, self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim,
                                             out_feats=self.hidden_dim,
                                             num_heads=self.heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             activation=self.activation,
                                             allow_zero_in_degree=True))
            self.mappings.append(thnn.Sequential(
                            thnn.Linear(2*self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim,self.hidden_dim,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.project = thnn.Linear(self.hidden_dim, self.n_classes,True)

    def forward(self, blocks, features, y_hat):
        layer=0
        label_embedding = self.label_embedding(y_hat)
        features = th.cat((features, label_embedding), dim=1)
        h = features
        for l in range(self.n_layers):
            if l==0:
                h = th.cat((self.layers[l](blocks[l], h).mean(1), h[:blocks[l].number_of_dst_nodes(), :]), dim=1)
                h = self.mappings[l](h)
            else:
                out = h
                h = th.cat((self.layers[l](blocks[l], h).mean(1), h[:blocks[l].number_of_dst_nodes(), :]), dim=1)
                h = self.mappings[l](h)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out[:blocks[l].number_of_dst_nodes()] + gate * h
                
            layer+=1
        
        h = self.bn(h)
        out = h
        h = self.MLP1(h)
        h = self.project(h)
        return h
    
class UnimpGraphConvModel(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout):
        super(UnimpGraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        emb_size = 64
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size, max_norm=True)
        self.mappings = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats+emb_size,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation,
                                          allow_zero_in_degree=True))
        self.mappings.append(thnn.Sequential(
                            thnn.Linear(self.in_feats+self.hidden_dim+emb_size,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
        for l in range(1, self.n_layers):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation,
                                              allow_zero_in_degree=True))
            self.mappings.append(thnn.Sequential(
                            thnn.Linear(2*self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)

    def forward(self, blocks, features, y_hat):
        layers=0
        label_embedding = self.label_embedding(y_hat)
        features = th.cat((features, label_embedding), dim=1)
        h = features
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            if layers == 0:
#                 h = gnn(block, h)
                h = th.cat((gnn(block, h), h[:block.number_of_dst_nodes(), :]), dim=1)
                h = self.mappings[l](h)
            else:
                out = h
#                 h = gnn(block, h)
                h = th.cat((gnn(block, h), h[:block.number_of_dst_nodes()]), dim=1)
                h = self.mappings[l](h)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out[:block.number_of_dst_nodes()] + gate * h
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            layers+=1
        h = self.bn(h)   
        out = h
        h = self.MLP1(h)
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h


class GraphSageModel(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0.1):
        super(GraphSageModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()
        self.filter_convs = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()

        # build multiple layers
        self.layers.append(dglnn.SAGEConv(in_feats=self.in_feats,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean'))
                                          # aggregator_type = 'pool'))
        for l in range(1, (self.n_layers)):
            self.layers.append(dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean'))
                                              # aggregator_type='pool'))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)
        
    def forward(self, blocks, features):
        h = features
        layers=0
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            if l == 0:
                h = gnn(block, h)
            else:
                out = h
                h = gnn(block, h)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out[:block.number_of_dst_nodes()] + gate * h
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            layers+=1
            
        h = self.bn(h)
        out = h
        h = self.MLP1(h)
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        # build multiple layers
        self.layers.append(dglnn.GraphConv(in_feats=self.in_feats,
                                           out_feats=self.hidden_dim,
                                           norm=self.norm,
                                           activation=self.activation,))
        for l in range(1, self.n_layers):
            self.layers.append(dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm,
                                               activation=self.activation))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)

    def forward(self, blocks, features):
        h = features
        layers=0
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            if layers == 0:
                h = gnn(block, h)
            else:
                out = h
                h = gnn(block, h)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out[:block.number_of_dst_nodes()] + gate * h
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
            layers+=1
        h = self.bn(h)   
        out = h
        h = self.MLP1(h)
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h


class GraphAttnModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop
                 ):
        super(GraphAttnModel, self).__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads
        self.feat_dropout = feat_drop
        self.attn_dropout = attn_drop
        self.activation = activation
        self.bn = thnn.BatchNorm1d(self.hidden_dim)
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        # build multiple layers
        self.layers.append(dglnn.GATConv(in_feats=self.in_feats,
                                         out_feats=self.hidden_dim,
                                         num_heads=self.heads[0],
                                         feat_drop=self.feat_dropout,
                                         attn_drop=self.attn_dropout,
                                         activation=self.activation))

        for l in range(1, self.n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(dglnn.GATConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                             out_feats=self.hidden_dim,
                                             num_heads=self.heads[l],
                                             feat_drop=self.feat_dropout,
                                             attn_drop=self.attn_dropout,
                                             activation=self.activation))
            self.gate_convs.append(thnn.Linear(self.hidden_dim*self.heads[l], 1))
        
        self.MLP1 = thnn.Sequential(
        nn.Linear(self.n_classes,self.n_classes,True),
        nn.BatchNorm1d(self.n_classes),
        nn.ELU()
        )
        self.MLP2 = thnn.Sequential(
        nn.Linear(self.n_classes,self.n_classes,True),
        nn.BatchNorm1d(self.n_classes),
        nn.ELU()
        )
        self.project = thnn.Linear(self.n_classes, self.n_classes,True)

    def forward(self, blocks, features):
        h = features
        layer = 0
        for l in range(self.n_layers - 1):
            if l==0:
                h = self.layers[l](blocks[l], h).flatten(1)
            else:
                out = h[:blocks[l].number_of_dst_nodes()]
                h = self.layers[l](blocks[l], h).flatten(1)
                gate = th.sigmoid(self.gate_convs[l-1](h))
                h = out + gate * h
            layer+=1
            
        h = self.layers[-1](blocks[-1],h).mean(1)
        layer+=1
        
        h = self.bn(h)
        out = h
        h = self.MLP1(h)
        h = 0.618**layer*h + out
        layer += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layer*h + out
        layer += 1
        
        h = self.project(h)
        return h
    
class RUNIMP(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0,
                rel_names=('cite','cite_by','self_loop'),
                aggregate='stack'):
        super(RUNIMP, self).__init__()
        self.rel_names = rel_names
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)
        self.compute_attn = thnn.ModuleList()
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        emb_size = 64
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size, max_norm=True)
        # build multiple layers
        self.r_bn = thnn.ModuleList()
#         myaggregate = 
        if aggregate=='stack':
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(in_feats=self.in_feats+emb_size,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean') for rel in self.rel_names}, aggregate=aggregate))
            _ = thnn.ModuleList()
            _.append(thnn.BatchNorm1d(self.hidden_dim))
            _.append(thnn.BatchNorm1d(self.hidden_dim))
            self.r_bn.append(_)
            self.compute_attn.append(thnn.Linear(self.hidden_dim, 1, True))
            for l in range(1, self.n_layers):
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean') for rel in self.rel_names}, aggregate=aggregate))
                _ = thnn.ModuleList()
                _.append(thnn.BatchNorm1d(self.hidden_dim))
                _.append(thnn.BatchNorm1d(self.hidden_dim))
                self.r_bn.append(_)
                self.compute_attn.append(thnn.Linear(self.hidden_dim, 1, True))
                self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        else:
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(in_feats=self.in_feats,
                                          out_feats=self.hidden_dim,
                                          aggregator_type='mean') for rel in self.rel_names}, aggregate=aggregate))
            for l in range(1, self.n_layers):
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.SAGEConv(in_feats=self.hidden_dim,
                                              out_feats=self.hidden_dim,
                                              aggregator_type='mean') for rel in self.rel_names}, aggregate=aggregate))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)
        
    def forward(self, blocks, features, y_hat):
        label_embedding = self.label_embedding(y_hat)
        features = {k: th.cat((v, label_embedding), dim=1) for k, v in features.items()}
        h = features
        
        layers=0
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            if l == 0:
                h = gnn(block, h)
#                 for i in range(2): # 2 relationships
#                     h['id'][:,i,:] = self.r_bn[l][i](h['id'][:,i,:])
                attns = {k: th.softmax(self.compute_attn[l](v), dim=-2) for k, v in h.items()}
                h = {k: th.sum(v*attns[k],dim=-2) for k, v in h.items()}
            else:
                out = h
                h = gnn(block, h)
#                 for i in range(2): # 2 relationships
#                     h[k][:,i,:] = self.r_bn[l][i](h[k][:,i,:])
                attns = {k: th.softmax(self.compute_attn[l](v), dim=-2) for k, v in h.items()}
                h = {k: th.sum(v*attns[k],dim=-2) for k, v in h.items()}
                h = {k: out[k][:block.number_of_dst_nodes()]+ th.sigmoid(self.gate_convs[l-1](v)) * v for k, v in h.items()}
    
            if l != len(self.layers) - 1:
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}
            layers+=1
        h = h['id']
        out = h
        h = self.MLP1(h)
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h
    
class HeteoGraphConvModel(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 dropout,
                 rel_names=('cite','cite_by'),
                 aggregate='stack'):
        super(HeteoGraphConvModel, self).__init__()
        self.rel_names = rel_names
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)
        self.norm = norm
        self.layers = thnn.ModuleList()
        self.gate_convs = thnn.ModuleList()
        emb_size = 64
        self.aggregate = aggregate
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size, max_norm=True)
        self.mappings = thnn.ModuleList()
        # build multiple layers
        
        # 可以对比stack， max, mean等方式等区别
        if aggregate=='stack':
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats=self.in_feats+self.hidden_dim,
                                               out_feats=self.hidden_dim//2,
                                               norm=self.norm) for rel in rel_names}, aggregate='stack'))
            self.mappings.append(thnn.Sequential(
                            thnn.Linear(self.in_feats+self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
            for l in range(1, self.n_layers):
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim//2,
                                               norm=self.norm) for rel in rel_names}, aggregate='stack'))
                self.mappings.append(thnn.Sequential(
                            thnn.Linear(2*self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
        else:
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats=self.in_feats+self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm) for rel in rel_names}, aggregate=aggregate))
            self.mappings.append(thnn.Sequential(
                            thnn.Linear(self.in_feats+self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
            for l in range(1, self.n_layers):
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GraphConv(in_feats=self.hidden_dim,
                                               out_feats=self.hidden_dim,
                                               norm=self.norm) for rel in rel_names}, aggregate=aggregate))
                self.mappings.append(thnn.Sequential(
                            thnn.Linear(2*self.hidden_dim,self.hidden_dim,True),
                            thnn.ELU(),
                            thnn.BatchNorm1d(self.hidden_dim)
                            ))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)
        
    def forward(self, blocks, features, y_hat):
        label_embedding = self.label_embedding(y_hat)
        features = {k: th.cat((v, label_embedding), dim=1) for k, v in features.items()}
        h = features
        layers=0
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            out = h
            tmp = gnn(block, h)
            tmp = {k: th.flatten(v, 1) for k, v in tmp.items()}
            h = {k: th.cat((tmp[k], h[k][:block.number_of_dst_nodes(), :self.in_feats]), dim=1) for k, v in tmp.items()}
            h = {k: self.mappings[l](v) for k, v in h.items()}
            if l!=0:
                h = {k: out[k][:block.number_of_dst_nodes()]+v for k, v in h.items()}
            if l != len(self.layers) - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
            
            layers+=1
        h = h['id']
        h = self.MLP1(h)
        
        h = self.project(h)
        return h

    
class HeteoGraphAttnModel(thnn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 activation=F.relu,
                 dropout=0,
                rel_names=('cite','cite_by','self_loop'),
                aggregate='stack'):
        super(HeteoGraphAttnModel, self).__init__()
        self.rel_names = rel_names
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.activation = activation
        self.dropout = thnn.Dropout(dropout)
        self.gate_convs = thnn.ModuleList()
        self.layers = thnn.ModuleList()
        emb_size = 64
        self.aggregate = aggregate
        self.label_embedding = thnn.Embedding(n_classes+1, emb_size, max_norm=True)
        self.mappings = thnn.ModuleList()
        self.compute_attn = thnn.ModuleList()

        
        # build multiple layers
        if aggregate=='stack':
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_feats=self.in_feats+emb_size,out_feats=self.hidden_dim,num_heads=1,feat_drop=.2,attn_drop=.2) for rel in self.rel_names}, aggregate='stack'))
            self.compute_attn.append(thnn.Linear(self.hidden_dim, 1, True))
        else:
            self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_feats=self.in_feats+emb_size,out_feats=self.hidden_dim,num_heads=1,feat_drop=.2,attn_drop=.2) for rel in self.rel_names}, aggregate=aggregate))
        self.mappings.append(thnn.Sequential(
                        thnn.Linear(self.in_feats+emb_size+self.hidden_dim, self.hidden_dim, True),
                        thnn.ELU(),
                        thnn.BatchNorm1d(self.hidden_dim)
                        ))
        for l in range(1, self.n_layers):
            if aggregate=='stack':
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_feats=self.hidden_dim,out_feats=self.hidden_dim,num_heads=1,feat_drop=.2,attn_drop=.2) for rel in self.rel_names}, aggregate=aggregate))
                self.compute_attn.append(thnn.Linear(self.hidden_dim, 1, True))
            else:
                self.layers.append(dglnn.HeteroGraphConv({rel: dglnn.GATConv(in_feats=self.hidden_dim,out_feats=self.hidden_dim,num_heads=1,feat_drop=.2,attn_drop=.2) for rel in self.rel_names}, aggregate=aggregate))
            self.mappings.append(thnn.Sequential(
                        thnn.Linear(2*self.hidden_dim,self.hidden_dim,True),
                        thnn.ELU(),
                        thnn.BatchNorm1d(self.hidden_dim)
                        ))
            self.gate_convs.append(thnn.Linear(self.hidden_dim, 1))
        
        self.MLP1 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )

        self.MLP2 = thnn.Sequential(
        thnn.Linear(self.hidden_dim, self.hidden_dim ,True),
        thnn.BatchNorm1d(self.hidden_dim),
        thnn.ELU()
        )
        
        self.project = thnn.Linear(self.hidden_dim, self.n_classes ,True)
        
    def forward(self, blocks, features, y_hat):
        label_embedding = self.label_embedding(y_hat)
        features = {k: th.cat((v, label_embedding), dim=1) for k, v in features.items()}
        h = features
        layers=0
        for l, (gnn, block) in enumerate(zip(self.layers, blocks)):
            out = h
            tmp = gnn(block, h)
            if self.aggregate=='stack':
                tmp = {k: th.squeeze(v) for k, v in tmp.items()}
                attns = {k: th.softmax(self.compute_attn[l](v), dim=-2) for k, v in tmp.items()}
                tmp = {k: th.sum(v*attns[k],dim=-2) for k, v in tmp.items()}
            tmp = {k: th.flatten(v, 1) for k, v in tmp.items()}
            h = {k: th.cat((tmp[k], h[k][:block.number_of_dst_nodes()]), dim=1) for k, v in tmp.items()}
            h = {k: self.mappings[l](v) for k, v in h.items()}
            if l!=0:
#                 h = {k: out[k][:block.number_of_dst_nodes()]+v for k, v in h.items()}
                h = {k: out[k][:block.number_of_dst_nodes()]+ th.sigmoid(self.gate_convs[l-1](v)) * v for k, v in h.items()}
                
            if l != len(self.layers) - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
            
            layers+=1
        
        out = h['id']
        h = self.MLP1(h['id'])
        h = 0.618**layers*h + out
        layers += 1

        out = h
        h = self.MLP2(h)
        h = 0.618**layers*h + out
        layers += 1
        
        h = self.project(h)
        return h
