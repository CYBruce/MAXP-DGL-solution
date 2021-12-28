# MAXP-DGL-solution([MAXP竞赛DGL图数据模型](https://biendata.xyz/competition/maxp_dgl/))
依赖包：
------
- dgl==0.7.1
- pytorch==1.7.0
- pandas
- numpy
- gensim

模型运行：
-------

1. 5个Jupyter Notebook文件使用Jupyter环境运行
2. 训练GNN模型

```bash
python get_n2v.py    # 获取node2vec embedding
# 运行unimp
python unimp_train.py --data_path ../final_data --gnn_model graphattn --hidden_dim 64 --n_layers 2 --fanout 20,20 --batch_size 4000 --GPU 0 --out_path ./results --epoch 40 --savename unimp_graphattn
```

## 比赛提分点

- 构建节点的embedding特征（如node2vec, deepwalk等）
- 图传播的label using
- 模型多折训练融合
- 在测试集推理时使用full sampler

## 已实验但无效trick

- 模型采用max bagging
- R-drop
- 特征降维
- post smoothig效果不好(参考c&s)
- 随机游走特征
- 伪标签训练

## 参考资料

1. Huang, Qian, et al. "Combining label propagation and simple models out-performs graph neural networks." *arXiv preprint arXiv:2010.13993* (2020).
2. Shi, Yunsheng, et al. "Masked label prediction: Unified message passing model for semi-supervised classification." *arXiv preprint arXiv:2009.03509* (2020).

