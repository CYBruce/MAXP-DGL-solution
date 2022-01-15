# MAXP-DGL-solution([MAXP竞赛DGL图数据模型](https://biendata.xyz/competition/maxp_dgl/))
## 任务背景

**任务**：图节点性质预测，即预测节点(论文)所属的类别。

**数据**：基于微软学术文献生成的论文关系图，其中的节点是论文，边是论文间的引用关系。包括约**150**万个节点，**2000**万条边。节点包含300维的特征来自论文的标题和摘要等内容。

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

## More than baseline

- 构建节点的embedding特征
- 模型多折训练融合


## 参考资料

1. Huang, Qian, et al. "Combining label propagation and simple models out-performs graph neural networks." *arXiv preprint arXiv:2010.13993* (2020).
2. Shi, Yunsheng, et al. "Masked label prediction: Unified message passing model for semi-supervised classification." *arXiv preprint arXiv:2009.03509* (2020).

