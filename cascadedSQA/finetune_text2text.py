# +
import torch
from transformers import ByT5Tokenizer, T5Tokenizer
import datasets

tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small',use_fast = False)
#tokenizer = T5Tokenizer.from_pretrained('t5-small',use_fast = False)

# process the examples in input and target text format and the eos token at the end 
def add_eos_to_examples(example):

    example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
    example['target_text'] = '%s </s>' % example['answers']['text'][0]
    
    return example

# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], padding='max_length', truncation=True, max_length=1024)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], padding='max_length', truncation=True, max_length=128)
    #print(input_encodings.keys())
    #print(target_encodings.keys())
    #s()
    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }
    print(encodings.keys())
    return encodings

# load train and validation split of squad
train_dataset  = datasets.load_dataset('squad', split="train")
valid_dataset = datasets.load_dataset('squad', split="validation")

# map add_eos_to_examples function to the dataset example wise 
train_dataset = train_dataset.map(add_eos_to_examples, load_from_cache_file=False)
# map convert_to_features batch wise
train_dataset = train_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
train_dataset.set_format(type="torch", columns=columns)
valid_dataset.set_format(type="torch", columns=columns)
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
def collate_batch(batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    #print(batch)
    #s()
    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['target_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
    

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'labels': lm_labels, 
        'decoder_attention_mask': decoder_attention_mask
      }
#ld = DataLoader(train_dataset,batch_size = 32,collate_fn = collate_batch)
#for b in ld:
    #print(b)
    #s()
print("training samples",len(train_dataset), "validation samples",len(valid_dataset))

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field

import json
import numpy as np
import torch

from transformers import T5ForConditionalGeneration, ByT5Tokenizer, EvalPrediction, T5Tokenizer
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

#from tpubar import TPUMonitor

#import torch_xla.core.xla_model as xm
#import torch_xla.distributed.xla_multiprocessing as xmp

logger = logging.getLogger(__name__)

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
#@dataclass
#class T2TDataCollator(DataCollator):



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    #make sure you save the arguments in at ./args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    #if xm.is_master_ordinal():
    #logger.warning(
    #    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #    training_args.local_rank,
    #    training_args.device,
    #    training_args.n_gpu,
    #    bool(training_args.local_rank != -1),
    #    training_args.fp16,
    #)
    #logger.info("Training/evaluation parameters %s", training_args)


    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # tokenizer = ByT5Tokenizer.from_pretrained(
    #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )

    tokenizer = ByT5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # Get datasets
    #train_dataset  = torch.load(data_args.train_file_path)
    #valid_dataset = torch.load(data_args.valid_file_path)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_batch,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

args_dict = {
  "num_cores": 8,
  'training_script': 'train_t5_squad.py',
  "model_name_or_path": 'voidful/phoneme_byt5', #'google/byt5-base',
  "tokenizer_name": 'google/byt5-small', #"google/byt5-base",
  "max_len": 1024 ,
  "target_max_len": 128,
  "output_dir": 't5lephone-small-squad',
  "overwrite_output_dir": True,
  "per_device_train_batch_size": 4,
  "per_device_eval_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "learning_rate": 3e-4,
  "num_train_epochs": 3,
  "do_train": True,
  "do_eval":False,
  "save_strategy": "epoch",
  "save_total_limit" : 100,
  "push_to_hub" : False,
  "remove_unused_columns" : False
  #"hub_token" : "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"
}
import os
os.environ["WANDB_DISABLED"] = "true"
with open('args.json', 'w') as f:
  json.dump(args_dict, f)
import torch.multiprocessing as mp

if __name__ == '__main__':
    #freeze_support()

    main()
    #mp.spawn(_mp_fn, args=(), nprocs=1, start_method='fork')
