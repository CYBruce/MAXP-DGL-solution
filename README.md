# MAXP-DGL-top3-solution([MAXP竞赛DGL图数据模型](https://biendata.xyz/competition/maxp_dgl/))
The solution code for MAXP DGL context (to be finished)

requirements：
------
- dgl==0.7.1
- pytorch==1.7.0
- pandas
- numpy

how to run：
-------
对于4个Jupyter Notebook文件，请使用Jupyter环境运行，并注意把其中的竞赛数据文件所在的文件夹替换为你自己保存数据文件的文件夹。
并记录下你处理完成后的数据文件所在的位置，供下面模型训练使用。


对于GNN的模型，需要先cd到gnn目录，然后运行：

对模型进行集成并输出结果
```bash
python merge.py
```

