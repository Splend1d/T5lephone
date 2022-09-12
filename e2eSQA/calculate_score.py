from datasets import load_dataset
splt = "test"
eval_dataset = load_dataset("Splend1dchan/NMSQA_hubert-l_features",split = splt)

df_hubert_count = eval_dataset.to_pandas()
df_hubert_count.rename(columns = {'question_file':'which_question'}, inplace = True)
eval_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str)

args = parser.parse_args()

out_path = args.outpath

def AOS(e):
    #qidx = int(e['which_question'])
    #ls = eval_dataset[qidx]["context_hubert_count"]
    m = min(e['predictions_start'],e['start_positions'])
    M = max(e['prediction_ends'],e['end_positions'])
    l = max(e['predictions_start'],e['start_positions'])
    r = min(e['prediction_ends'],e['end_positions'])
    #print(r,l,M,m)
    #print((r-l)/(M-m))
    #s()
    if r-l <= 0:
        return 0
    q = e["question_len"]
    r_context = r-q-1
    l_context = l-q-1
    M_context = M-q-1
    m_context = m-q-1
    #return (e["acc_cnt"][r_context] - e["acc_cnt"][max(l_context-1,0)]) / (e["acc_cnt"][M_context] - e["acc_cnt"][max(m_context-1,0)])
    return (r-l+1)/(M-m+1)
    #return output
def FF1(e):
    #qidx = int(e['which_question'])
    #ls = eval_dataset[qidx]["context_hubert_count"]
    m = min(e['predictions_start'],e['start_positions'])
    M = max(e['prediction_ends'],e['end_positions'])
    l = max(e['predictions_start'],e['start_positions'])
    r = min(e['prediction_ends'],e['end_positions'])
    x_delta = e['prediction_ends'] - e["predictions_start"] + 1
    y_delta = e['end_positions'] - e["start_positions"] + 1
    if (r-l) >= 0:
        precision = (r-l+1)/(x_delta)
        recall = (r-l+1)/(y_delta)
        ff1 = 2 * precision * recall / (precision + recall)
    else:
        ff1 = 0
    #print(r,l,M,m)
    #print((r-l)/(M-m))
    #s()
    
    return ff1


import pandas as pd
import numpy as np
import tqdm
df = pd.read_csv(out_path)
print(df.columns)
#df = df.merge(df_hubert_count[["which_question","context_hubert_count"]],how = "right",on= "which_question")
#acc_cnt = []
#for ls in df["context_hubert_count"]:
#    acc_cnt.append(np.cumsum(np.array(ls)))
#df["acc_cnt"] = acc_cnt
df["AOS"] = df.apply(AOS, axis = 1)
df["FF1"] = df.apply(FF1, axis = 1)
#for row in df.iterrows():
#    print(row)
print(len(list(df["which_question"].unique())))
df

df.head()
len(df)

df["index"] = range(len(df))
df["which_question2"] = list(df["which_question"])
df["confidences2"] = list(df["confidences"])
#df["which_question_real"] = df["index"]//1000*1000 + df["which_question"]

df = df.drop(columns = ["context_hubert_count","acc_cnt"])
df

# +
df2 = df.groupby(['confidences','which_question']).max()
#df2 = df2.groupby(['which_question']).max()
#df2 = df2[df2["start_positions"]!=0]


df3 = df2[['confidences2','which_question2']].groupby(['which_question2']).max()
df3["maxconfidence"] = list(df3["confidences2"])
df2 = df2.merge(df3[["maxconfidence"]], how='right', on='which_question2')

df2 = df2[df2["confidences2"] == df2["maxconfidence"]]
# -

df2

print(df2["AOS"].mean()) #0.5125465841220072

print(df2["FF1"].mean())

print("SQuAD AOS/FF1")
print(df2[df2["which_question2"].str.contains("squad")]["AOS"].mean())

print(df2[df2["which_question2"].str.contains("squad")]["FF1"].mean())

print("Non SQuAD AOS/FF1")
print(df2[~df2["which_question2"].str.contains("squad")]["AOS"].mean())

print(df2[~df2["which_question2"].str.contains("squad")]["FF1"].mean())




