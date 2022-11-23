from datasets import load_dataset
#splt = "test"




import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str)

args = parser.parse_args()

import os
out_path = args.out_path
out_dir = os.path.dirname(out_path)
import re
epoch = int(out_path.split("checkpoint-")[1].split("_")[0])
splt = "dev" if "dev" in out_path else "test"
print("EVALUATING ON",splt)
eval_dataset = load_dataset("Splend1dchan/NMSQA_hubert-l_features",split = splt)
df_hubert_count = eval_dataset.to_pandas()
df_hubert_count.rename(columns = {'question_file':'which_question'}, inplace = True)
 #s()

if "t5" in out_path:
    OFFSET = 1
else: # longformer
    OFFSET = 3


def AOS(e):
    #qidx = int(e['which_question'])
    #ls = eval_dataset[qidx]["context_hubert_count"]
    #print(e)
    #s()
    #print(e)
    #print(e["question_len"])
    #print(len(e["question_hubert_count"]))
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
    r_context = r-q-OFFSET
    l_context = l-q-OFFSET
    M_context = M-q-OFFSET
    m_context = m-q-OFFSET
    #assert r_context >= 0
    #assert l_context >= 0
    #assert M_context >= 0
    #assert m_context >= 0
    #print(r_context,l_context,M_context,m_context)
    r = e["acc_cnt"][r_context]
    if l_context == 0:
        l = 0
    else:
        l = e["acc_cnt"][l_context]
    M = e["acc_cnt"][M_context]
    if m_context == 0:
        m = 0
    else:
        m = e["acc_cnt"][m_context]
   
    return (r-l)/(M-m)
    #return output
def FF1(e):
    #qidx = int(e['which_question'])
    #ls = eval_dataset[qidx]["context_hubert_count"]
    #print(e)
    m = min(e['predictions_start'],e['start_positions'])
    M = max(e['prediction_ends'],e['end_positions'])
    l = max(e['predictions_start'],e['start_positions'])
    r = min(e['prediction_ends'],e['end_positions'])
    if r-l <= 0:
        return 0
    #print(m,M,l,r)
    q = e["question_len"]
    r_context = r-q-OFFSET
    l_context = l-q-OFFSET
    M_context = M-q-OFFSET
    m_context = m-q-OFFSET
    #print(M_context,m_context)
    
    r = e["acc_cnt"][r_context]
    if l_context == 0:
        l = 0
    else:
        l = e["acc_cnt"][l_context]
    M = e["acc_cnt"][M_context]
    if m_context == 0:
        m = 0
    else:
        m = e["acc_cnt"][m_context]
    #print(r,l,M,m)
    x_delta = r-l
    y_delta = M-m
    if (r-l) >= 0:
        precision = (r-l)/(x_delta)
        recall = (r-l)/(y_delta)
        ff1 = 2 * precision * recall / (precision + recall)
    else:
        ff1 = 0
    #
    #print((r-l)/(M-m))
    #s()
    assert ff1 <= 1
    return ff1


import pandas as pd
import numpy as np
import tqdm
df = pd.read_csv(out_path)
#print(df.columns)
df = df.merge(df_hubert_count[["which_question","context_hubert_count"]],how = "right",on= "which_question")
acc_cnt = []
for ls in df["context_hubert_count"]:
    acc_cnt.append(np.cumsum(np.array(ls)))
df["acc_cnt"] = acc_cnt
df["AOS"] = df.apply(AOS, axis = 1)
df["FF1"] = df.apply(FF1, axis = 1)
#for row in df.iterrows():
#    print(row)
#print(len(list(df["which_question"].unique())))
df

df.head()
len(df)

df["index"] = range(len(df))
df["which_question2"] = list(df["which_question"])
df["confidences2"] = list(df["confidences"])
#df["which_question_real"] = df["index"]//1000*1000 + df["which_question"]

df = df.drop(columns = ["context_hubert_count","acc_cnt"])
#df

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



f = open(f"{out_dir}/score_{splt}.txt","a")
print("EPOCH",epoch,file = f)
print("SQuAD AOS/FF1", file = f)
print(df2[df2["which_question2"].str.contains("squad")]["AOS"].mean(), file = f)
print(df2[df2["which_question2"].str.contains("squad")]["FF1"].mean(), file = f)
print("Non SQuAD AOS/FF1", file = f)
print(df2[~df2["which_question2"].str.contains("squad")]["AOS"].mean(), file = f)
print(df2[~df2["which_question2"].str.contains("squad")]["FF1"].mean(), file = f)
print("",file = f)
