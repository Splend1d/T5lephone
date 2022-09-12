# +
import torch
from transformers import LongformerTokenizerFast, AutoModelForQuestionAnswering, AutoConfig, T5Config
from t5qa import T5ForQuestionAnswering
from enc_t5 import EncT5ForQuestionAnswering, EncLongT5ForQuestionAnswering
from enc_led import EncLEDForQuestionAnswering
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os 
import json 

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from torch.nn import LogSoftmax

# ./models/longformer-base-4096/checkpoint-5950
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--checkpoint', type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--max_seq_length', default=1024, type=int)
parser.add_argument('--doc_stride', default=256, type=int)
parser.add_argument('--splt', default="dev", type=str) # dev or test
#parser.add_argument('--output_fname', default="eval_verbose-2975", type=str)
args = parser.parse_args()
#enc-t5lephone-small-1024
#enc-byt5-small-1024/checkpoint-31700
#longformer-base-4096/checkpoint-8925

# +
if "long-t5" in args.model_name_or_path:
    model = EncLongT5ForQuestionAnswering.from_pretrained(args.model_name_or_path).cuda()
elif "t5" in args.model_name_or_path:
    model = EncT5ForQuestionAnswering.from_pretrained(args.model_name_or_path).cuda()
elif "led" in args.model_name_or_path:
    model = EncLEDForQuestionAnswering.from_pretrained(args.model_name_or_path).cuda()
else:
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path).cuda()
    
model.eval()
# -

'''
post-processing the answer prediction
'''
def _get_best_indexes(probs,context_offset, n_best_size):
    """Get the n-best logits from a list."""
    best_indexes = torch.topk(probs[context_offset:],n_best_size).indices + context_offset
    return best_indexes


def post_process_prediction(start_prob, end_prob, context_offset, n_best_size=10, max_answer_length=500, weight=0.6):
    prelim_predictions = []
    start_prob = start_prob.squeeze()
    end_prob = end_prob.squeeze()
    #input_id = input_id.squeeze()
    
    start_indexes = _get_best_indexes(start_prob, context_offset, n_best_size)
    end_indexes = _get_best_indexes(end_prob, context_offset, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant
    #print("best indexes",start_indexes, end_indexes)
    #print(start_indexes,end_indexes)
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            #if start_index >= len(input_id):
            #    continue
            #if end_index >= len(input_id):
            #    continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue

            predict = {
                        'start_prob': start_prob[start_index],
                        'end_prob': end_prob[end_index],
                        'start_idx': start_index, 
                        'end_idx': end_index, 
                      }

            prelim_predictions.append(predict)

    prelim_predictions = sorted(prelim_predictions, 
                                key=lambda x: (x['start_prob'] + x['end_prob']), 
                                reverse=True)
    
    if len(prelim_predictions) > 0:
        final_start_idx = prelim_predictions[0]['start_idx'].item()
        final_end_idx = prelim_predictions[0]['end_idx'].item()
        confidence = (prelim_predictions[0]['start_prob'] + prelim_predictions[0]['end_prob']).item()
    else:
        final_start_idx = torch.argmax(start_prob).cpu().item()
        final_end_idx = torch.argmax(end_prob).cpu().item()
        confidence = -10000
    return final_start_idx, final_end_idx, confidence


question_column_name = "question_hubert_code"
context_column_name = "context_hubert_code"
answer_start_column_name = "code_start"
answer_end_column_name = "code_end"
#config = AutoConfig.from_pretrained("./models/longformer-base-4096/checkpoint-2975")
#print(config)
max_seq_length = args.max_seq_length
GLOBAL_BATCH = 0
def prepare_validation_features(examples):
    global GLOBAL_BATCH
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    if "t5" in args.model_name_or_path or "led" in args.model_name_or_path:
        examples[question_column_name] = [[x+3 for x in ls] for ls in examples[question_column_name]]
        examples[context_column_name] = [[x+3 for x in ls] for ls in examples[context_column_name]]
        qlens = [len(q) for q,c in zip(examples[question_column_name], examples[context_column_name])]
        clens = [len(c) for q,c in zip(examples[question_column_name], examples[context_column_name])]
    else:
        examples[question_column_name] = [[x+5 for x in ls] for ls in examples[question_column_name]]
        examples[context_column_name] = [[x+5 for x in ls] for ls in examples[context_column_name]]
        qlens = [len(q) for q,c in zip(examples[question_column_name], examples[context_column_name])]
        clens = [len(c) for q,c in zip(examples[question_column_name], examples[context_column_name])]
    qlens = [len(q) for q,c in zip(examples[question_column_name], examples[context_column_name])]
    clens = [len(c) for q,c in zip(examples[question_column_name], examples[context_column_name])]

    tokenized_examples = {}
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["input_ids"] = []
    tokenized_examples["attention_mask"] = []
    ii = 0
    tokenized_examples["which_question"]  = []
    tokenized_examples["question_len"] = []
    for qlen,clen, ans_start, ans_end in zip(qlens,clens, examples[answer_start_column_name], examples[answer_end_column_name]):

        if "t5" in args.model_name_or_path or "led" in args.model_name_or_path:
            print("using <s> </s>")
            #q<\s>c1<\s> c1_len : 4096 - 2 - q, c1_start = 0 => ans = 1 + len(q) + 1 + 0
            #q<\s>c2<\s> c2_start : c1_len - (args.doc_stride) => ans = 1 + len(q) + 1 + len(c1) - (args.doc_stride) 
            nxstart = 0
            c_span_len = max_seq_length - 2 - qlen
            is_pad = False
            while True:

                start = nxstart
                end = start + c_span_len -1
                #print(ii,start,end)
                nxstart = end + 1 - (args.doc_stride)
                seq = examples[question_column_name][ii]+[1]+examples[context_column_name][ii][start:end+1]+[1]
                attn = [1] * len(seq)
                if len(seq) < max_seq_length:
                    seq = seq + (max_seq_length - len(seq)) * [0]
                    tokenized_examples["input_ids"].append(seq[:])
                    attn += (max_seq_length - len(attn)) * [0]
                    tokenized_examples["attention_mask"].append(attn[:])
                    is_pad = True
                else:
                    tokenized_examples["input_ids"].append(seq[:])
                    tokenized_examples["attention_mask"].append(attn[:])

                if ans_start >= start and ans_end <= end:
                    tokenized_examples["start_positions"].append(1 + qlen + ans_start-start)
                    tokenized_examples["end_positions"].append(1 + qlen + ans_end-start)
                    #print(qlen,tokenized_examples["start_positions"][-1])
                    #print(qlen,tokenized_examples["end_positions"][-1],start,ans_end)
                    assert tokenized_examples["start_positions"][-1] < max_seq_length
                    assert tokenized_examples["end_positions"][-1] < max_seq_length 
                else: #not is this span
                    tokenized_examples["start_positions"].append(0)
                    tokenized_examples["end_positions"].append(0)
                tokenized_examples["which_question"].append(examples["question_file"][ii])
                tokenized_examples["question_len"].append(qlens[ii])

                if is_pad:
                    break
        else:
            #<s>q<\s><\s>c1<\s> c1_len : 4096 - 4 - q, c1_start = 0 => ans = 1 + len(q) + 1 + 1 + 0
            #<s>q<\s><\s>c2<\s> c2_start : c1_len - (args.doc_stride) => ans = 1 + len(q) + 1 + 1 + len(c1) - (args.doc_stride) 

            nxstart = 0
            c_span_len = max_seq_length - 4 - qlen
            is_pad = False
            while True:

                start = nxstart
                end = start + c_span_len - 1
                #print(ii,start,end)
                nxstart = end + 1 - (args.doc_stride)
                seq = [0]+examples[question_column_name][ii]+[2]+[2]+examples[context_column_name][ii][start:end+1]+[2]
                attn = [1] * len(seq)
                if len(seq) < max_seq_length:
                    seq = seq + (max_seq_length - len(seq)) * [1]
                    tokenized_examples["input_ids"].append(seq[:])
                    attn += (max_seq_length - len(attn)) * [0]
                    tokenized_examples["attention_mask"].append(attn[:])
                    is_pad = True
                else:
                    tokenized_examples["input_ids"].append(seq[:])
                    tokenized_examples["attention_mask"].append(attn[:])

                if ans_start >= start and ans_end <= end:
                    tokenized_examples["start_positions"].append(1 + qlen + 1 + 1 + ans_start-start)
                    tokenized_examples["end_positions"].append(1 + qlen + 1 + 1 + ans_end-start)
                    assert tokenized_examples["start_positions"][-1] < max_seq_length
                    assert tokenized_examples["end_positions"][-1] < max_seq_length 
                else: #not is this span
                    tokenized_examples["start_positions"].append(0)
                    tokenized_examples["end_positions"].append(0)
                tokenized_examples["which_question"].append(examples["question_file"][ii])
                tokenized_examples["question_len"].append(qlens[ii])

                if is_pad:
                    break
            

            
        ii += 1
    print(len(tokenized_examples["question_len"]))
    
    GLOBAL_BATCH += 1
    print("GLOBAL", GLOBAL_BATCH)
    return tokenized_examples


def collate_dev_fn(batch):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    input_ids = torch.LongTensor([example['input_ids'] for example in batch])
    attention_mask = torch.LongTensor([example['attention_mask'] for example in batch])
    start_positions = torch.LongTensor([example['start_positions'] for example in batch])
    end_positions = torch.LongTensor([example['end_positions'] for example in batch])
    question_len = torch.LongTensor([example['question_len'] for example in batch])
    
    return {
        'input_ids': input_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'attention_mask': attention_mask, 
        'question_len': question_len, 
    }



# +
from datasets import load_dataset

eval_dataset = load_dataset("Splend1dchan/NMSQA_hubert-l_features",split = args.splt)
column_names = eval_dataset.column_names
print(column_names)
eval_dataset = eval_dataset.map(
    prepare_validation_features,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on validation dataset",
)
dataloader = DataLoader(eval_dataset, batch_size=16,collate_fn = collate_dev_fn, shuffle=False, num_workers = 4)
# -


print(len(eval_dataset))
from tqdm import tqdm

prediction_starts = []
prediction_ends = []
confidences = []
with torch.no_grad():
    for n,batch in tqdm(enumerate(dataloader),total = len(dataloader)):
        #print(batch['input_ids'].shape)
        #print()
        outputs = model(input_ids=batch['input_ids'].cuda(),attention_mask=batch['attention_mask'].cuda())
        #prediction_starts += torch.argmax(outputs.start_logits,dim = 1).cpu().tolist()
        #prediction_ends += torch.argmax(outputs.end_logits,dim = 1).cpu().tolist()
        for j in range(outputs.start_logits.shape[0]):
            kshot = 1
            confidence = -10000
            while confidence == -10000:
                prediction_start, prediction_end, confidence = post_process_prediction(outputs.start_logits[j], outputs.end_logits[j],batch['question_len'][j]+2,kshot,256)
                kshot += 2
                #print(kshot)
            prediction_starts.append(prediction_start)
            prediction_ends.append(prediction_end)
            confidences.append(confidence)
        if n == 10:
            pass
            #break




out_path = os.path.join(args.output_dir, f"out-{args.splt}_checkpoint-{args.checkpoint}_len-{args.max_seq_length}.csv")
eval_dataset.to_csv(out_path, index = False)
eval_csv = pd.read_csv(out_path)
eval_csv = eval_csv.drop(columns = ["input_ids","attention_mask"])
eval_csv["predictions_start"] = prediction_starts + [0] * (len(eval_csv)-len(prediction_starts))
eval_csv["prediction_ends"] = prediction_ends + [0] * (len(eval_csv)-len(prediction_ends))
eval_csv["confidences"] =  confidences + [0] * (len(eval_csv)-len(prediction_ends))
eval_csv.to_csv(out_path, index = False)


