from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor
import soundfile as sf
from datasets import load_dataset
import torchaudio
import torch
import pickle as pkl
import os
from tqdm import tqdm
import librosa 

model_name = "wav2vec2-large-10min-lv60-self"
header = "facebook" if "960" in model_name else "Splend1dchan"
processor = AutoProcessor.from_pretrained(f"{header}/{model_name}")

model = AutoModelForCTC.from_pretrained(f"{header}/{model_name}").cuda()
model_cpu = AutoModelForCTC.from_pretrained(f"{header}/{model_name}")
model_cpu.eval()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(f"{header}/{model_name}")
feature_extractor = AutoFeatureExtractor.from_pretrained(f"{header}/{model_name}")

res = {}
fold = "dev"
paths = sorted(os.listdir(f"NMSQA_audio/{fold}_audios"))
paths_set = set(paths)

def get_asr(idx):
  global res, paths
  
  
  start = idx
  while idx < len(paths):
    file = paths[idx]
    #print(file)
    if not file.endswith("mp3"):
      idx += 1
      continue
    elif file.replace("mp3","lab") not in paths_set:
      idx += 1
      continue
    fpath = os.path.join(f"NMSQA_audio/{fold}_audios",file)
    speech, sampling_rate = torchaudio.load(fpath)
    speech = librosa.resample(speech.squeeze().numpy(), orig_sr=sampling_rate, target_sr=16000)
    print(idx,fpath,speech.shape[-1],len(res))
    if speech.shape[-1] > 22050 * 35:
      input_values = processor(speech, return_tensors="pt", padding="longest",sampling_rate = 16000).input_values.squeeze(1)
      logits = model_cpu(input_values).logits
    else:
      input_values = processor(speech, return_tensors="pt", padding="longest",sampling_rate = 16000).input_values.squeeze(1).cuda()  # Batch size 1
      
      logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_ids = pred_ids.squeeze()

    outputs = tokenizer.decode(pred_ids, output_word_offsets=True)

    time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

    word_offsets = [

      {

          "word": d["word"],

          "start_time": round(d["start_offset"] * time_offset, 2),

          "end_time": round(d["end_offset"] * time_offset, 2),

      }

      for d in outputs.word_offsets

    ]
    res[file] = {"word_offsets":word_offsets,"asr":outputs["text"]}
    paths = sorted(os.listdir(f"NMSQA_audio/{fold}_audios"))
    # sometimes there will be unexpected errors
    # to save progress, save every 1000 items
    if idx % 1000 == 0:
      filehandler = open(f"asr_results/{model_name}/{fold}-{start}-{idx}.pkl","wb")
      pkl.dump(res,filehandler)
      filehandler.close()
      res = {}
    idx += 1
    del speech, input_values,pred_ids

  filehandler = open(f"asr_results/{model_name}/{fold}.pkl","wb")
  pkl.dump(res,filehandler)
get_asr(0)