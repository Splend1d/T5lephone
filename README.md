# T5lephone
Code for T5lephone: Bridging Speech and Text Self-supervised Models for Spoken Language Understanding via Phoneme level T5

## Pretraining T5lephone
```
https://github.com/voidful/t5lephone
```
1. Obtain raw text corpus such as wiki

2. Phonemize using espeak 

3. Map espeak characters to ASCII characters

4. Generate span reconstruction files

5. Run pretraining (Span Reconstruction Obejective)
python pretraining.py
You may change initialize model from T5, ByT5, LongT5 (Generative model recommended)

**! Alternatively you can use our checkpoints**
```
from byt5 - voidful/phoneme_byt5
from mt5 - voidful/phoneme_mt5
from longt5 - voidful/phoneme_longt5
```


## Cascaded SQA
```
go to ./cascadedSQA
```

### ASR Generation

1. download dataset from voidful/NMSQA

unzip the tar files so that the data has the following structure
```
--NMSQA_audio
 |--train_audios
 |--dev_audios
 |--test_audios
```


2. run gen_ASR.py to generate ASR results
```
python gen_ASR.py 
```
ASR results (.pkl files) will be saved under ```asr_results/{speech_model_name}```



3. run gen_ASRparquet.py
```
python gen_ASRparquet.py
```
this turns the pickle file into a parquet file, which could be uploaded to the huggingface dataset

Following process will utilize the data from using ```load_dataset``` method in huggingface


**! Alternatively you can skip ASR generation by downloading the preprocessed sample from huggingface**
```
# question_normalized_text column : ASR results
# question_times : timespan for each word in the ASR results
ASR results from wav2vec2-large-960h-lv60-self : Splend1dchan/NMSQA_wav2vec2-large-960h-lv60-self
ASR results from wav2vec2-large-10min-lv60-self : Splend1dchan/NMSQA_wav2vec2-large-10min-lv60-self
```
### Training on SQuAD

for extractive models (longformer, deberta), follow huggingface ```run_squad.py```

for generative models (T5 + variants), run ```finetune_text2text.py``` with ```transformers==4.18.0```

subword T5 parameters -- "max_len" : 512, "target_max_len": 16

character T5 parameters -- "max_len" : 1024, "target_max_len": 128

**! Alternatively, you can use our (and other people's) checkpoint on huggingface to skip this step**
```
longformer : valhalla/longformer-base-4096-finetuned-squadv1
deberta : Palak/microsoft_deberta-large_squad
T5-small : Splend1dchan/t5-small-squad
T5-base : valhalla/t5-base-squad
T5-large : Splend1dchan/t5-large-squad
Byt5-base : Splend1dchan/byt5-base-squad
Byt5-small : Splend1dchan/byt5small-squad1024-from6000steps
Byt5lephone-small : Splend1dchan/t5lephone-small-textsquad

```


### Evaluate on SQuAD
for generative models run ``` evaluate_NMSQA.ipynb ```
code is tested on GPU google colab environment

**! This produces the AOS/FF1 scores in the paper**

## End-to-End SQA

### Prepare HuBERT Features
follow https://github.com/DanielLin94144/DUAL-textless-SQA

**! Alternatively, you can use prepared huggingface dataset ```Splend1dchan/NMSQA_hubert-l_features```**
These columns are provided in the dataset
```
question_hubert_code : hubert clusters after reducing repetitive elements
context_hubert_code : hubert clusters after reducing repetitive elements
question_hubert_cnt : number of consecutive hubert clusters before reducing repetitive elements
context_hubert_cnt : number of consecutive hubert clusters before reducing repetitive elements

ex : question_hubert_code = [52,91] and question_hubert_cnt = [3,4]
=> original sequence = [52,52,52,91,91,91,91]

```
### Training



```bash
# Change max_seq_len according to model
python -m torch.distributed.launch \
  --nproc_per_node=8 transformers/examples/pytorch/question-answering/run_qa_nmsqa.py \
  --model_name_or_path {language model name} \
  --dataset_name Splend1dchan/NMSQA_hubert-l_features \
  --dataloader_num_workers 4 \
  --do_train \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_step 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --warmup_steps 500 \
  --logging_steps 50 \
  --max_seq_length 4096 \
  --doc_stride 256 \
  --save_strategy "epoch" \
  --ddp_find_unused_parameters=True \
  --output_dir {output dir} 
 ```

**! Models of length <= 512 is not supported, as some questions exceed this length which results in the data preprocessing being frozen**

ByT5-small setting : google/byt5-small, max_length = 1024
ByT5lephone setting : voidful/phoneme_byt5_g2p_v1, max length = 1024
longformer setting : allenai/longformer-base-4096, max_length = 1024 or 4096

### Evaluation 

1. Generate Predictions
```bash
# Change max_seq_len according to model
python evaluate_e2eSQA.py \
  --model_name_or_path {language model finetuned on e2eSQA} \
  --output_dir {output dir for predictions} \
  --checkpoint {number of steps of the finetuned language model (int)} \ 
```

2. Calculate AOS/FF1

```bash
# Change max_seq_len according to model
python evaluate_e2eSQA.py \
  --out_path {output csv from 1.}
```
Score will be printed on screen

## Speech Translation

### Training
```bash
bash run_e2eST.sh
```
or
```bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
 train.py --speech_model_config facebook/wav2vec2-large-lv60 \
--nlp_model_config google/byt5-small \
--nlp_model_decoder_only \
--SpeechMixEEDT5 \
--dataset google/xtreme_s \
--field covost2.en.de \
--filter \
--train_split train \
--test_split validation \
--batch 8 \
--grad_accum 1 \
--epoch 30 \
--worker 4 \
--share_layer_ratio 0 \
--down_scale 8 \
--eval_step  4502 \
--lr 4e-5 \
--warmup_steps 700 \
--wandb \
--fixed_parameters True \
--notes wav2vec2-large-lv60_byt5-small_textdecoderonly_bs64

```
**! Alternatively, you can use our checkpoint on huggingface to skip this step**

All intermediate checkpoints could be found in the commits
```
wav2vec2-l -> byt5-small :
Splend1dchan/wav2vec2-large-lv60_byt5-small_textdecoderonly_bs64

wav2vec2-l -> byt5lephone-small :
Splend1dchan/wav2vec2-large-lv60_t5lephone-small_textdecoderonly_bs64

wav2vec2-l -> mt5-small :
Splend1dchan/wav2vec2-large-lv60_mt5-small_textdecoderonly_bs64

wav2vec2-l -> mt5lephone-small :
Splend1dchan/wav2vec2-large-lv60_mt5lephone-small_textdecoderonly_bs64

```
### Evaluation
1. Generate Prediction

**! It is advised to generate while training (it is also quite time consuming), so intermediate checkpoints are not lost**
```bash
n=4502
while true
do
    FILE=/work/twskvfb446/facebook/wav2vec2-large-lv60_google/byt5-base_SpeechMixEEDT5_wav2vec2-large-lv60_byt5-base_textdecoderonly_bs64/checkpoint-${n}
    if [ -f "${FILE}/pytorch_model.bin" ]; then
        echo "running eval ${n}"
        python3 eval_hf_fast.py --speech_model_config facebook/wav2vec2-large-lv60 \
        --nlp_model_config google/byt5-small \
        --nlp_model_decoder_only \
        --SpeechMixEEDT5eval \
        --pretrain_path ${FILE} \
        --dataset google/xtreme_s \
        --field covost2.en.de \
        --train_split train \
        --test_split validation \
        --batch 8 \
        --grad_accum 8 \
        --epoch 30 \
        --worker 32 \
        --share_layer_ratio 0 \
        --down_scale 8 \
        --lr 4e-5 \
        --warmup_steps 500 \
        --wandb \
        --fixed_parameters True \
        --notes w2v2-large_byt5-small
        
        let "n+=4502" 
    else
        echo "NA waiting for $n"
        sleep 60
    fi
done

```
2. Calculate Score

```python

from collections import defaultdict
import pandas as pd
import datasets
step = 4502
epoch = 30
scores = {}
scores["byt5"] = defaultdict(list)

for checkpoint in range(step,step*epoch+1,step):
    pretrain_path = f"facebook/wav2vec2-large-lv60_google/byt5-small_SpeechMixEEDT5_wav2vec2-large-lv60_byt5-small_textdecoderonly_bs64/predictions/predictions-{checkpoint}_fromhf_largebatch.csv"
    dfout = pd.read_csv(pretrain_path)
    sacrebleu = datasets.load_metric("sacrebleu")
    predictions = [pred for pred in dfout["preds"]]
    references = [[gold] for gold in dfout["gold"]]
    results = sacrebleu.compute(predictions=predictions, references=references)
    #print(results["score"])
    
    print(checkpoint/step,checkpoint,{"bleu":results["score"]})
    scores["byt5"]["epoch"].append(checkpoint/step)
    scores["byt5"]["score"].append(results["score"])

```
