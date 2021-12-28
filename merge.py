import numpy as np
import pandas as pd

results_list = [
'./finals/logits_unimp_graphattn_n2v1_fold0.npz',
'./finals/logits_unimp_graphattn_n2v1_fold1.npz',
'./finals/logits_unimp_graphattn_n2v1_fold2.npz',
'./finals/logits_unimp_graphattn_n2v1_fold3.npz',
'./finals/logits_unimp_graphattn_n2v1_fold4.npz',
'./finals/logits_unimp_graphattn_n2v1_fold5.npz',
'./finals/logits_unimp_graphattn_n2v1_fold6.npz',
'./finals/logits_unimp_graphattn_n2v1_fold7.npz',
'./finals/logits_unimp_graphattn_n2v1_fold8.npz',
'./finals/logits_unimp_graphattn_n2v1_fold9.npz',
'./finals/logits_unimp2_graphattn_fold0.npz',
'./finals/logits_unimp2_graphattn_fold1.npz'
]
weights = [1] * len(results_list)
sum_logits = 0

# 多折融合
for i in range(len(weights)):
    _ = np.load(results_list[i])
    idx, logits = _['ids'], _['logits']
    sum_logits+=(logits * weights[i])
# np.save('./finals/results/output.npy', sum_logins)     
# 已经保存了运行的结果，可直接得到提交结果
with open('./finals/results/output.npy', 'rb') as f:
    sum_logits =np.load(f, allow_pickle=True)

    
output_df = pd.DataFrame(sum_logits)
output_df['node_idx'] = idx
nodes_df = pd.read_csv("./final_data/IDandLabels.csv")
output_df = output_df.merge(nodes_df, how='left', left_on='node_idx', right_on='node_idx')

merge_output = output_df[list(np.arange(23))].values 
merge_output=merge_output.argmax(-1)
merge_output = [chr(ord('A') + int(x)) for x in list(merge_output)]
output_df = output_df[['paper_id']]
output_df.columns = [['id']]
output_df['label'] = merge_output
output_df.to_csv('output.csv', index=False)