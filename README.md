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
python get_n2v.py
python unimp_train.py
```


## 参考资料

1. Shi, Yunsheng, et al. "Masked label prediction: Unified message passing model for semi-supervised classification." *arXiv preprint arXiv:2009.03509* (2020).
2. Wang, Yangkun, et al. "Bag of tricks for node classification with graph neural networks." *arXiv preprint arXiv:2103.13355* 2.3 (2021).  

