from datasets import load_dataset, Audio
from transformers import AutoModelForPreTraining, AutoProcessor, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import pandas as pd

# _tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

encoder_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60", sampling_rate = 16000)
def collate_batch(batch: List):
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
        A dictionary of tensors
    """
    #print(batch[0].keys())
    speech_input = [example['speech_input'] for example in batch]
    input_values = encoder_processor(speech_input, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values
    #print(speech_input)
    text_prompt_ids = torch.stack([example['text_prompt_ids'] for example in batch])
    #print(len(speech_input))
    #print(text_prompt_ids.shape)
    
    #print(text_prompt_ids)
    #text_prompt_attention_mask = torch.stack([example['text_prompt_attention_mask'] for example in batch])

    #lm_labels = torch.stack([example['target_ids'] for example in batch])
    

    labels = torch.stack([example['target_ids'] for example in batch])
    decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
    #print(labels_batch)

    labels[labels[:, :] == 0] = -100
    #print(labels.shape)
    #print(decoder_attention_mask.shape)
    #s()
    return {
        'input_values': input_values,
        "decoder_text_prompt_ids":text_prompt_ids,
        'labels': labels, 
        'decoder_attention_mask': decoder_attention_mask
      }

class STDataset(Dataset):
    def __init__(self,splt = "train",tokenizer = "google/mt5-small",translate_from="en", translate_to="de", size = None, _filter = True):
        self.data = load_dataset("google/xtreme_s", f"covost2.{translate_from}.{translate_to}",cache_dir = "./",revision="1.0.0")[splt]
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16_000))
        print(self.data)
        #print(self.data[0]["audio"]["array"])
        #s()
        self.translate_from = translate_from
        self.translate_to = translate_to
        if "mt5" in tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.decode_max_len = 256 if "byt" in tokenizer else 64
        print(self.decode_max_len)

        input_text_prompt = f"Translate {self.translate_from} -> {self.translate_to}: "
        input_text_prompt_tensors = self.tokenizer(input_text_prompt, return_tensors = "pt")
        self.text_prompt_ids = input_text_prompt_tensors["input_ids"]
        self.text_prompt_attention_mask = input_text_prompt_tensors["attention_mask"]

        if _filter and splt =="train":
            df_filter = pd.read_csv(f"metadata/covost2.{translate_from}.{translate_to}/{splt}_parsed.csv")
            self.select = [n for n,x in enumerate(df_filter["filter1"]) if x == 1]
        else:
            self.select = list(range(len(self.data)))
        #self.l = [len(self.data[i]["translation"].split()) for i in range(len(self.data))]
        #print(max(self.l))
        #s()
        if size is None:
            self.size = len(self.select)
        else:
            self.size = size


    def __len__(self):
        return self.size

    def __getitem__(self,ii):
        idx = self.select[ii]
        #input_values, input_text_prompt=None,decoder_input_ids=None, labels=None
        speech_input = self.data[idx]["audio"]["array"]
        #print(speech_input.shape)
        #sr = self.data[idx]["audio"]["sampling_rate"]
        #print(sr)
        #s()
        text_output = self.data[idx]["translation"]
        text_output_tensors = self.tokenizer(
            text_output, 
            return_tensors = "pt", 
            padding = "max_length", 
            max_length = self.decode_max_len,
            truncation = "longest_first",
        )
        target_ids = text_output_tensors["input_ids"].squeeze()
        target_attention_mask = text_output_tensors["attention_mask"].squeeze()

        #lm_labels = torch.stack([example['target_ids'] for example in batch])
        #lm_labels[lm_labels[:, :] == 0] = -100
        #attention_mask = torch.stack([example['attention_mask'] for example in batch])

        #decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

         
        return {
            "speech_input":speech_input,
            "text_prompt_ids":self.text_prompt_ids.squeeze(),
            "target_ids":target_ids,
            "target_attention_mask":target_attention_mask,
        }

class STDatasetSmall(Dataset):
    def __init__(self,splt = "train",tokenizer = "google/mt5-small", translate_from="en", translate_to="de"):
        self.data = load_dataset("google/xtreme_s", f"covost2.{translate_from}.{translate_to}",cache_dir = "./",revision="1.0.0")[splt]
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16_000))
        self.translate_from = translate_from
        self.translate_to = translate_to
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.decode_max_len = 256 if "byt" in tokenizer else 64
        print(self.decode_max_len)

        input_text_prompt = f"Translate {self.translate_from} -> {self.translate_to}: "
        input_text_prompt_tensors = self.tokenizer(input_text_prompt, return_tensors = "pt")
        self.text_prompt_ids = input_text_prompt_tensors["input_ids"]
        self.text_prompt_attention_mask = input_text_prompt_tensors["attention_mask"]

        if splt =="train":
            df_filter = pd.read_csv(f"metadata/covost2.{translate_from}.{translate_to}/{splt}_parsed.csv")
            self.select = [n for n,x in enumerate(df_filter["filter1"]) if x == 1]
        else:
            self.select = list(range(len(self.data)))
        #self.l = [len(self.data[i]["translation"].split()) for i in range(len(self.data))]
        #print(max(self.l))
        #s()


    def __len__(self):
        return len(self.select[:12800])

    def __getitem__(self,ii):
        idx = self.select[ii]
        #input_values, input_text_prompt=None,decoder_input_ids=None, labels=None
        speech_input = self.data[idx]["audio"]["array"]
        #sr = self.data[idx]["audio"]["sampling_rate"]
        #print(sr)
        #s()
        text_output = self.data[idx]["translation"]
        text_output_tensors = self.tokenizer(
            text_output, 
            return_tensors = "pt", 
            padding = "max_length", 
            max_length = self.decode_max_len,
            truncation = "longest_first",
        )
        target_ids = text_output_tensors["input_ids"].squeeze()
        target_attention_mask = text_output_tensors["attention_mask"].squeeze()

        #lm_labels = torch.stack([example['target_ids'] for example in batch])
        #lm_labels[lm_labels[:, :] == 0] = -100
        #attention_mask = torch.stack([example['attention_mask'] for example in batch])

        #decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

         
        return {
            "speech_input":speech_input,
            "text_prompt_ids":self.text_prompt_ids.squeeze(),
            "target_ids":target_ids,
            "target_attention_mask":target_attention_mask,
        }


class STDatasetMinds(Dataset):
    def __init__(self,splt = "train",tokenizer = "google/mt5-small", size = None, trim = 320000):
        self.data = load_dataset("google/xtreme_s", "minds14.en-US", cache_dir = "./")[splt]
        self.data2 = load_dataset("google/xtreme_s", "minds14.en-GB", cache_dir = "./")[splt]
        self.data3 = load_dataset("google/xtreme_s", "minds14.en-AU", cache_dir = "./")[splt]
        self.ndata = [len(self.data),len(self.data2), len(self.data3)]
        self.data = self.data.cast_column("audio", Audio(sampling_rate=16_000))
        self.data2 = self.data.cast_column("audio", Audio(sampling_rate=16_000))
        self.data3 = self.data.cast_column("audio", Audio(sampling_rate=16_000))
        self.label = [self.data.features["intent_class"].names[self.data[x]["intent_class"]] for x in range(len(self.data))]
        self.label += [self.data.features["intent_class"].names[self.data2[x]["intent_class"]] for x in range(len(self.data))]
        self.label += [self.data.features["intent_class"].names[self.data3[x]["intent_class"]] for x in range(len(self.data))]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.decode_max_len = 25 if "byt" in tokenizer else 10
        print(self.decode_max_len)

        input_text_prompt = f"Intent Classification: "
        input_text_prompt_tensors = self.tokenizer(input_text_prompt, return_tensors = "pt")
        self.text_prompt_ids = input_text_prompt_tensors["input_ids"]
        self.text_prompt_attention_mask = input_text_prompt_tensors["attention_mask"]
        
        self.intent_str = self.data.features["intent_class"].names
        
        self.select = list(range(len(self.data)+len(self.data2)+len(self.data3)))
        #self.l = [len(self.data[i]["translation"].split()) for i in range(len(self.data))]
        #print(max(self.l))
        #s()
        if size is None:
            self.size = len(self.select)
        else:
            self.size = size
        
        self.trim = trim


    def __len__(self):
        return self.size

    def __getitem__(self,ii):
        idx = self.select[ii]
        if idx < len(self.data):
            choice = self.data
        elif idx < len(self.data) + len(self.data2):
            choice = self.data2
            idx -= len(self.data)
        else:
            choice = self.data3
            idx -= len(self.data) + len(self.data2)
            
        #input_values, input_text_prompt=None,decoder_input_ids=None, labels=None
        speech_input = choice[idx]["audio"]["array"]
        if self.trim:
            if speech_input.shape[-1] > self.trim:
                speech_input = speech_input[:self.trim]
        #print(speech_input.shape)
        #print(speech_input.shape)
        #sr = self.data[idx]["audio"]["sampling_rate"]
        #print(sr)
        #s()
        text_output = self.intent_str[choice[idx]["intent_class"]]
        text_output_tensors = self.tokenizer(
            text_output, 
            return_tensors = "pt", 
            padding = "max_length", 
            max_length = self.decode_max_len,
            truncation = "longest_first",
        )
        target_ids = text_output_tensors["input_ids"].squeeze()
        target_attention_mask = text_output_tensors["attention_mask"].squeeze()

        #lm_labels = torch.stack([example['target_ids'] for example in batch])
        #lm_labels[lm_labels[:, :] == 0] = -100
        #attention_mask = torch.stack([example['attention_mask'] for example in batch])

        #decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

         
        return {
            "speech_input":speech_input,
            "text_prompt_ids":self.text_prompt_ids.squeeze(),
            "target_ids":target_ids,
            "target_attention_mask":target_attention_mask,
        }

if __name__ == "__main__":
    translate_from = "en"
    translate_to = "de"
    train_dataset = STDatasetForMetaData("train",translate_from = "en",translate_to = "de")
    #train_loader = DataLoader(train_dataset, batch_size = 3, collate_fn = collate_batch)
    max_length = 0
    from tqdm import tqdm
    metadata = {"index":[],"speech_input_len":[],"source":[],"target":[]}
    for n,element in tqdm(enumerate(train_dataset),total = len(train_dataset)):
        metadata["index"].append(n)
        metadata["speech_input_len"].append(element["speech_input"])
        metadata["source"].append(element["text_output_tgt"])
        metadata["target"].append(element["text_output_src"])
        
        #metadata["source"].append(train_dataset.data[0]["transcription"])
        #metadata["target"].append(train_dataset.data[0]["translation"])
        #metadata["target_tokenized_len"].append(element["target_ids"].shape[0])
        #break
        #print()
        #print(element["target_ids"].shape)
        if n % 1000 == 0:
            df = pd.DataFrame.from_dict(metadata)
            df.to_csv(f"./metadata/covost2.{translate_from}.{translate_to}/train.csv",index = False)
    df = pd.DataFrame.from_dict(metadata)
    df.to_csv(f"./metadata/covost2.{translate_from}.{translate_to}/train.csv",index = False)
    print(max_length)
    for element in train_loader:
        print(element)
        break
    print(max_length)

# encoder_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")#
# encoder_model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60").cuda()#
# print(encoder_model.config)
# s()

# covost_2 = load_dataset("google/xtreme_s", "covost2.en.de",cache_dir = "./")
# #covost2.en.de-data_dir=.%2Fdata
# audio_input = covost_2["train"][0]["audio"]["array"]  # first decoded audio sample
# inputs = encoder_processor(audio_input, return_tensors="pt", padding="longest")
# input_values = inputs.input_values.to("cuda")
# attention_mask = inputs.attention_mask.to("cuda")

# with torch.no_grad():
#     out = encoder_model(input_values, attention_mask=attention_mask)
#     print(out.projected_states.shape)
#     print(out.shape)
# s()
# transcription = covost_2["train"][0]["transcription"]  # first transcription

# translation = covost_2["train"][0]["translation"]  # first translation
# print(transcription)
# print(translation)

