import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import pandas as pd

import asrp
import torch
import torchaudio
from datasets import load_dataset, Audio, load_from_disk
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, TrainerCallback, \
    TrainerState, TrainerControl

import speechmix
import datasets

from dataset import STDataset, collate_batch, STDatasetSmall
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize

def create_self_decoder_input(decoder_model, tokenizer, input_sent, device):
    gen_input = tokenizer(input_sent, add_special_tokens=True).input_ids
    predicted = [decoder_model.config.decoder_start_token_id]
    with torch.no_grad():
        decoder_model.eval()
        decoder_length = max(decoder_model.config.max_length, len(gen_input))
        for _ in range(decoder_length):
            max_item = torch.argmax(
                decoder_model(input_ids=torch.tensor([gen_input], device=device),
                              output_hidden_states=True,
                              decoder_input_ids=torch.tensor(
                                  [predicted],
                                  device=device)).logits, -1)[:, -1].item()
            if decoder_model.config.eos_token_id == max_item:
                break
            predicted.append(max_item)
    return gen_input, predicted[1:]


def main(arg=None):
    def prepare_dataset_custom(batch):
        path = batch["path"]
        speech, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)
        batch["input_values"] = resampler.forward(speech.squeeze(0)).numpy()
        batch["length"] = len(batch["input_values"])
        batch["labels"] = model.tokenizer(batch["text"]).input_ids
        return batch

    def prepare_dataset(batch, selftype=False):
        audio = batch["audio"]
        new_batch = {}
        new_batch["input_values"] = audio["array"]
        new_batch["length"] = audio["array"].size
        sent = batch["text"] if 'text' in batch else batch["sentence"]
        sent = sent.lower()
        if selftype:
            input_text_prompt = "ASR: "
            decoder_input, decoder_target = create_self_decoder_input(model.decoder_model, model.tokenizer,
                                                                      input_text_prompt + sent,
                                                                      model.device)
            new_batch['input_text_prompt'] = input_text_prompt
            new_batch["text_input_ids"] = decoder_input
            new_batch['labels'] = decoder_target
        else:
            gen_input = model.tokenizer(sent, add_special_tokens=False).input_ids
            new_batch['labels'] = gen_input
        new_batch['labels'] += [model.tokenizer.eos_token_id]
        return new_batch

    def prepare_dataset_speechtranslation(batch, translate_from = "En", translate_to = "De"):
        audio = batch["audio"]
        new_batch = {}
        new_batch["input_values"] = audio["array"]
        new_batch["length"] = audio["array"].size
        sent = batch["translation"]
        sent = sent.lower()
        #if selftype:
        input_text_prompt = f"Translate {translate_from} -> {translate_to}: "
        decoder_input, decoder_target = create_self_decoder_input(model.decoder_model, model.tokenizer,
                                                                  input_text_prompt + sent,
                                                                  model.device)
        new_batch['input_text_prompt'] = input_text_prompt
        new_batch["text_input_ids"] = decoder_input
        new_batch['labels'] = decoder_target
        #else:
        #    gen_input = model.tokenizer(sent, add_special_tokens=False).input_ids
        #    new_batch['labels'] = gen_input
        new_batch['labels'] += [model.tokenizer.eos_token_id]
        return new_batch

    def compute_metrics(pred):
        pred_ids = pred.predictions
        pred_ids = [i[i != -100] for i in pred_ids]
        #print(pred_ids[0])
        pred_str = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
        #print(pred_str[0])
        # we do not want to group tokens when computing the metrics
        label_ids = pred.label_ids
        label_ids = [i[i != -100] for i in label_ids]
        label_str = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
        #print(label_str)
        # for l, p in zip(label_str, pred_str):
        #     print(l, "======", p)
        cer = asrp.cer(label_str, pred_str)
        wer = asrp.wer(label_str, pred_str)
        return {"cer": cer, "wer": wer}

    @dataclass
    class DataCollatorWithPadding:
        tokenizer: AutoTokenizer
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None
        selftype: bool = False

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            batch = {}
            batch['input_values'] = [torch.tensor(feature["input_values"]) for feature in features]
            label_features = [{"input_ids": feature['labels']} for feature in features]
            labels_batch = self.tokenizer.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

            if 'text_input_ids' in features[0]:
                text_features = [{"input_ids": feature['text_input_ids']} for feature in features]
                text_batch = self.tokenizer.pad(
                    text_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )
                batch['text_input_ids'] = text_batch['input_ids']
            labels_batch = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyway
            if self.tokenizer.bos_token_id and (labels_batch[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
                labels_batch = labels_batch[:, 1:]

            batch['labels'] = labels_batch

            torch.cuda.empty_cache()
            return batch

    class FreezingCallback(TrainerCallback):
        def __init__(self, trainer, freeze_model, freeze_epoch=3):
            self.trainer = trainer
            self.freeze_model = freeze_model
            self.freeze_epoch = freeze_epoch
            self.current_step_idx = 0
            self.default_param_fix = {}
            self.name_list = []
            for name, param in self.freeze_model.named_parameters():
                self.name_list.append(name)
                self.default_param_fix[name] = param.requires_grad
            self.freeze_layers = int(len(self.default_param_fix.keys()) / freeze_epoch)

        def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.epoch < self.freeze_epoch:
                release = self.name_list[-int(self.freeze_layers * state.epoch):]
                for name, param in self.freeze_model.named_parameters():
                    if name in release:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in self.freeze_model.named_parameters():
                    param.requires_grad = self.default_param_fix[name]
            self.current_step_idx += 1

        def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            for name, param in self.trainer.model.named_parameters():
                param.requires_grad = True

    def parse_args(args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--speech_model_config", type=str)
        parser.add_argument("--nlp_model_config", type=str)
        parser.add_argument("--SpeechMixEED", action='store_true')
        parser.add_argument("--SpeechMixEEDT5", action='store_true')
        parser.add_argument("--SpeechMixEEDT5eval", action='store_true')
        parser.add_argument("--SpeechMixED", action='store_true')
        parser.add_argument("--SpeechMixSelf", action='store_true')
        parser.add_argument("--SpeechMixAdapter", action='store_true')
        parser.add_argument("--SpeechMixGAN", action='store_true')
        parser.add_argument("--SpeechMixFixed", action='store_true')
        parser.add_argument("--HFSpeechMixEED", action='store_true')
        parser.add_argument("--HFSpeechMixED", action='store_true')
        parser.add_argument("--HFSpeechMixSelf", action='store_true')
        parser.add_argument("--HFSpeechMixAdapter", action='store_true')
        parser.add_argument("--HFSpeechMixGAN", action='store_true')
        parser.add_argument("--HFSpeechMixFixed", action='store_true')
        parser.add_argument("--pretrain_path",type = str)
        parser.add_argument("--cache", action='store_true')
        parser.add_argument("--dataset", type=str)
        parser.add_argument("--field", type=str)
        parser.add_argument("--train_split", type=str)
        parser.add_argument("--test_split", type=str)
        parser.add_argument("--notes", type=str)
        parser.add_argument("--grad_accum", default=3, type=int)
        parser.add_argument("--logging_steps", default=10, type=int)
        parser.add_argument("--warmup_steps", default=500, type=int)
        parser.add_argument("--unfreeze_warmup_steps", default=1000, type=int)
        parser.add_argument("--save_total_limit", default=2, type=int)
        parser.add_argument("--max_grad_norm", default=10, type=int)
        parser.add_argument("--worker", default=10, type=int)
        parser.add_argument("--batch", type=int)
        parser.add_argument("--epoch", default=1000, type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--eval_step", default=700, type=int)
        parser.add_argument('--share_layer_ratio', default=0, type=float)
        parser.add_argument('--down_scale', default=8, type=int)
        parser.add_argument('--weighted_sum', action='store_true')
        parser.add_argument('--fixed_parameters', action='store_true')
        parser.add_argument("--custom_set", type=str)
        parser.add_argument('--fixed_except', nargs='+',
                            default=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                                     "layernorm_embedding", 'attention', 'encoder'])
        parser.add_argument("--fp16", action='store_true')
        parser.add_argument("--wandb", action='store_true')

        input_args, model_arg = parser.parse_known_args(args)
        input_args = {k: v for k, v in vars(input_args).items() if v is not None}
        other_arg = {k.replace("--", ""): v for k, v in zip(model_arg[:-1:2], model_arg[1::2])}
        return input_args, other_arg

    input_args, other_arg = parse_args(sys.argv[1:]) if arg is None else parse_args(arg)
    print("input_args", input_args)


    if input_args['SpeechMixEED']:
        model_type = "SpeechMixEED"
        model = speechmix.SpeechMixEED(**input_args)

    elif input_args['SpeechMixEEDT5']:
        model_type = "SpeechMixEEDT5"
        model = speechmix.SpeechMixEEDT5(**input_args)
    elif input_args['SpeechMixEEDT5eval']:
        model_type = "SpeechMixEEDT5eval"
        model = speechmix.SpeechMixEEDT5eval(**input_args)
    elif input_args['SpeechMixFixed']:
        model_type = "SpeechMixFixed"
        model = speechmix.SpeechMixFixed(**input_args)
    elif input_args['SpeechMixSelf']:
        model_type = "SpeechMixSelf"
        model = speechmix.SpeechMixSelf(**input_args)
    elif input_args['SpeechMixGAN']:
        model_type = "SpeechMixGAN"
        model = speechmix.SpeechMixGAN(**input_args)
    elif input_args['SpeechMixAdapter']:
        model_type = "SpeechMixAdapter"
        model = speechmix.SpeechMixAdapter(**input_args)
    elif input_args['HFSpeechMixED']:
        model_type = "HFSpeechMixED"
        model = speechmix.HFSpeechMixED(**input_args)
    elif input_args['HFSpeechMixEED']:
        model_type = "HFSpeechMixEED"
        model = speechmix.HFSpeechMixEED(**input_args)
    elif input_args['HFSpeechMixFixed']:
        model_type = "HFSpeechMixFixed"
        model = speechmix.HFSpeechMixFixed(**input_args)
    elif input_args['HFSpeechMixSelf']:
        model_type = "HFSpeechMixSelf"
        model = speechmix.HFSpeechMixSelf(**input_args)
    elif input_args['HFSpeechMixGAN']:
        model_type = "HFSpeechMixGAN"
        model = speechmix.HFSpeechMixGAN(**input_args)
    elif input_args['HFSpeechMixAdapter']:
        model_type = "HFSpeechMixAdapter"
        model = speechmix.HFSpeechMixAdapter(**input_args)
    else:
        model_type = "SpeechMixEED"
        model = speechmix.SpeechMixEED(**input_args)
    print("model_done")
    
    #model.from_pretrained(input_args['pretrain_path'])
    state_dict = torch.load(os.path.join(input_args['pretrain_path'],"pytorch_model.bin"))
    model.load_state_dict(state_dict)
    #print(state_dict)
    #s()
    # selftype = ('SpeechMixSelf' in model_type or 'SpeechMixGAN' in model_type)
    selftype = 'SpeechMixSelf' in model_type
    if 'custom_set' in input_args:
        cache_file_train = f"{input_args['custom_set']}_hf_train.data"
        if input_args['cache'] and os.path.isdir(cache_file_train):
            train_ds = load_dataset('csv', data_files=input_args['custom_set'])['train']
            train_ds = train_ds.load_from_disk(cache_file_train)
            print("train_ds", train_ds)
        else:
            dataset = load_dataset('csv', data_files=input_args['custom_set'], cache_dir='./.cache')
            dataset = dataset['train']
            dataset = dataset.train_test_split(test_size=0.1)
            train_ds = dataset['train']
            train_ds = train_ds.map(prepare_dataset_custom, keep_in_memory=False, num_proc=input_args["num_proc"])
            train_ds.save_to_disk(cache_file_train)

        cache_file_test = f"{input_args['custom_set']}_hf_test.data"
        if input_args['cache'] and os.path.isdir(cache_file_test):
            valid_ds = load_dataset('csv', data_files=input_args['custom_set'])['train']
            valid_ds = valid_ds.load_from_disk(cache_file_test)
        else:
            dataset = load_dataset('csv', data_files=input_args['custom_set'], cache_dir='./.cache')
            dataset = dataset['train']
            dataset = dataset.train_test_split(test_size=0.1)
            valid_ds = dataset['test']
            valid_ds = valid_ds.map(prepare_dataset_custom, keep_in_memory=False, num_proc=input_args["num_proc"])
            valid_ds.save_to_disk(cache_file_test)
        print(train_ds)
    else:
        cache_path_train = f'./train_ds_{input_args["dataset"]}_{model_type}_{input_args["speech_model_config"]}_{input_args["nlp_model_config"]}_{input_args["field"]}_{input_args["train_split"]}.parquet'
        cache_path_valid = f'./valid_ds_{input_args["dataset"]}_{model_type}_{input_args["speech_model_config"]}_{input_args["nlp_model_config"]}_{input_args["field"]}_{input_args["train_split"]}.parquet'
        if input_args['cache'] and os.path.exists(cache_path_train) and os.path.exists(cache_path_valid):
            train_ds = load_from_disk(cache_path_train)
            valid_ds = load_from_disk(cache_path_valid)
        else:
            #train_ds = STDataset(splt=input_args["train_split"])
            valid_ds = STDataset(splt=input_args["test_split"], tokenizer=input_args["nlp_model_config"])
            #train_ds = load_dataset(input_args["dataset"], input_args["field"], split=input_args["train_split"], cache_dir = "./")
            #valid_ds = load_dataset(input_args["dataset"], input_args["field"], split=input_args["test_split"], cache_dir = "./")

            #train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
            #valid_ds = valid_ds.cast_column("audio", Audio(sampling_rate=16_000))

            #train_ds = train_ds.map(prepare_dataset_speechtranslation, num_proc=8)
            #valid_ds = valid_ds.map(prepare_dataset_speechtranslation, num_proc=8)

            #train_ds.save_to_disk(cache_path_train)
            #valid_ds.save_to_disk(cache_path_valid)

    data_collator = collate_batch

    training_args = TrainingArguments(
        output_dir=f"./tmp",
        per_device_train_batch_size=int(input_args['batch']),
        per_device_eval_batch_size=int(input_args['batch']),
        gradient_accumulation_steps=int(input_args['grad_accum']),
        eval_accumulation_steps=2,
        optim="adafactor",
        evaluation_strategy="steps",
        # group_by_length=True,
        load_best_model_at_end=True,
        fp16=input_args.get('fp16', True),
        num_train_epochs=input_args.get('epoch', 10),
        save_steps=input_args.get('eval_step', 700),
        eval_steps=input_args.get('eval_step', 700),
        logging_steps=input_args.get('logging_steps', 10),
        learning_rate=input_args.get('lr', 5e-4),
        warmup_steps=input_args.get('warmup_steps', 500),
        save_total_limit=input_args.get('save_total_limit', 2),
        dataloader_num_workers=input_args.get('worker', 10),
        report_to="wandb" if input_args.get('wandb', True) else "none",
    )
    print(training_args)
    # some trainer problem - save all logistics on compute_metrics, cause out of memory, fix:argmax first;
    # dynamic padding on past key value, cause index error, fix: return only loss and logist
    # group_by_length took lots of time during preprocessing
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=valid_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    # https://discuss.huggingface.co/t/gradual-layer-freezing/3381/4
    freezing_callback = FreezingCallback(trainer, model.encoder_model, input_args.get('unfreeze_warmup_steps', 500))
    trainer.add_callback(freezing_callback)

    predictions = trainer.predict(valid_ds)
    pred_ids = predictions.predictions
    pred_ids = [i[i != -100] for i in pred_ids]
    #print(pred_ids[0])
    pred_str = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, group_tokens=False)
    #print(ret)
    df = pd.read_csv("metadata/covost2.en.de/validation.csv")

    out= {"gold":list(df["translation"])[:len(valid_ds)],"preds":pred_str}
    print("Sample:",out["gold"][0],out["preds"][0])
    dfout = pd.DataFrame().from_dict(out)
    dfout.to_csv(input_args['pretrain_path'].replace("checkpoint","predictions")+"csv",index = False)
    
    predictions = [pred for pred in out["preds"]]
    references = [[gold] for gold in out["gold"]]

    sacrebleu = datasets.load_metric("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)
    print(results["score"])
    # bleus = []
    # for gold,pred in zip(out["gold"],out["preds"]):
    #     gold = word_tokenize(gold)
    #     pred = word_tokenize(pred)
    #     b1 = sentence_bleu([gold], pred, weights=(1, 0, 0, 0))
    #     b2 = sentence_bleu([gold], pred, weights=(0, 1, 0, 0))
    #     b3 = sentence_bleu([gold], pred, weights=(0, 0, 1, 0))
    #     b4 = sentence_bleu([gold], pred, weights=(0, 0, 0, 1))
    #     bavg4 = sentence_bleu([gold], pred, weights=(0.25, 0.25, 0.25, 0.25))
    #     bleus.append(sentence_bleu([gold], pred, weights=(0.25, 0.25, 0.25, 0.25)))
    #     print(b1,b2,b3,b4,bavg4)
    # avg_bleu = sum(bleus)/len(bleus)
    cer = asrp.cer(out["gold"], out["preds"])
    wer = asrp.wer(out["gold"], out["preds"])
    print({"cer": cer, "wer": wer, "bleu":results["score"]})

if __name__ == "__main__":
    main()
