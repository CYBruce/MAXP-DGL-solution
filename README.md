# MAXP-DGL-solution([MAXP竞赛DGL图数据模型](https://biendata.xyz/competition/maxp_dgl/))
## 任务背景

**任务**：图节点性质预测，即预测节点(论文)所属的类别。

**数据**：基于微软学术文献生成的论文关系图，其中的节点是论文，边是论文间的引用关系。包括约**150****万个节点**，**2000****万条边**。节点包含300维的特征来自论文的标题和摘要等内容。

**难点**：数据规模大，半监督学习，图神经网络。

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

## 比赛中的尝试

### 数据层面

✕ 利用PCA进行特征降维

✕ 进行样本增强，具体方法为对同一分类的样本互换特征，但保留邻居信息

### 模型层面

✕ Correct & smooth，模型欠拟合

? 构建引用和被引用的异构图并做Heterographconv，训练慢且无明显收效

? R-drop：利用dropout的随机性，训练时做两次prediction，然后loss项增加两次prediction的divergence

### 参数层面

✕ 修改各类Loss function，收益不明显

✕ 第一层采样数多于第二层采样数

✕ Label using时加入噪声

? 增加模型层数（性价比低）

### 后处理

✕ 模型融合采用max bagging（和投票区别不大）

✕ post smoothig效果不好（可能因为无标签的数据占比较高）

✕ 伪标签重训练（无标签的数据占比较高且模型本身准确率偏低）

✕ GAT，GCN，graphsage模型融合（精确度偏低）

✕ 融合R-unimp结果

## 参考资料

1. Huang, Qian, et al. "Combining label propagation and simple models out-performs graph neural networks." *arXiv preprint arXiv:2010.13993* (2020).
2. Shi, Yunsheng, et al. "Masked label prediction: Unified message passing model for semi-supervised classification." *arXiv preprint arXiv:2009.03509* (2020).

