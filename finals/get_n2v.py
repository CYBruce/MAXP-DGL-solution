import pandas as pd
import numpy as np
import os
import dgl
import os
from gensim.models import Word2Vec
import argparse

parser = argparse.ArgumentParser(description='DGL_SamplingTrain')
parser.add_argument('--p', type=int, default=1)
parser.add_argument('--q', type=int, default=1)
parser.add_argument('--walk_length', type=int, default=10)
parser.add_argument('--window', type=int, default=5)
parser.add_argument('--EMB_SIZE', type=int, default=64)
parser.add_argument('--savename', type=str, default='n2v.npy')
args = parser.parse_args()


graphs, _ = dgl.load_graphs(os.path.join('../final_data', 'graph.bin'))
graph = graphs[0]

# graph = dgl.to_bidirected(graph, copy_ndata=True)
graph = dgl.add_self_loop(graph)
print('################ Graph info: ###############')

print(np.mean(graph.in_degrees().numpy()))
print(np.mean(graph.out_degrees().numpy()))

EMB_SIZE = args.EMB_SIZE
nodes = []
for repeat in range(10):
    nodes.extend([i for i in range(graph.num_nodes())])

print("generating random walks")
walks = dgl.sampling.node2vec_random_walk(graph, nodes, args.p, args.q, walk_length=args. walk_length)
print("sampling complete")
walks = walks.numpy().tolist()
print("training")
model = Word2Vec(walks, vector_size=EMB_SIZE, window=args.window, min_count=0, epochs=50, sg=1, workers=16)
print("training complete")
w2v = np.zeros([len(model.wv.index_to_key), EMB_SIZE])
print(w2v.shape)
for i, index_id in enumerate(sorted(model.wv.index_to_key)):
    w2v[i] = model.wv[int(index_id)]

with open(os.path.join('../final_data', args.savename), 'wb') as f:
    np.save(f, w2v)


