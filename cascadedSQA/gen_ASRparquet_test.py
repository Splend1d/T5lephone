import os
import pickle as pkl

import librosa
import pandas as pd
import torchaudio

df = pd.read_parquet('./NMSQA/data/test-00000-of-00001-e59cc4b2d3e13fe2.parquet', engine='pyarrow')
df = df.dropna(axis=0, how='any', subset=["content_full_audio_path", "question_audio_path"])
df = df.reset_index(drop=True)

model_name = "wav2vec2-large-10min-lv60-self"
splt = "test"
basic = "word"
asr = {}
for fname in os.listdir(f"asr_results/{model_name}"):
    if fname.startswith(f"{splt}") and fname.endswith("pkl"):
        print(fname)
        file = open(f"asr_results/{model_name}/{fname}", 'rb')

        asr_partial = pkl.load(file)
        asr.update(asr_partial)
print(asr.keys())
print(asr[list(asr.keys())[0]])

err = 0
content_times = []
content_transcriptions = []
question_times = []
question_transcriptions = []
from tqdm import tqdm

for i in tqdm(range(len(df))):
    for col in ["content_full_audio_path", "question_audio_path"]:

        if col == "content_full_audio_path":
            assert df[col][i] != None
            files = sorted([f for f in asr if df[col][i][:-4] + "_" in f and f.endswith(".mp3")],
                           key=lambda x: int(x.split("-")[-1].split(".")[0]))
            print(files)
            # s()
            time = []
            transcription = ""
            offset = 0
            for f in files:
                speech, sample_rate = torchaudio.load(os.path.join("NMSQA_audio/test_audios", f))
                speech = librosa.resample(speech.squeeze().numpy(), orig_sr=sample_rate, target_sr=16000)
                duration = speech.shape[-1] / 16000
                for w in asr[f][f"{basic}_offsets"]:
                    time.append([(w["start_time"] + offset), (w["end_time"] + offset)])
                    if len(time) > 2:
                        if not (time[-1][0] >= time[-2][1]):
                            print(time[-2:])
                            # s()
                transcription += asr[f]["asr"]
                transcription += " "
                offset += duration

            content_times.append(time)
            content_transcriptions.append(transcription)
        elif col == "question_audio_path":
            files = [df[col][i]]
            print(files)
            time = []
            transcription = ""
            offset = 0
            for f in files:

                speech, sample_rate = torchaudio.load(os.path.join("NMSQA_audio/test_audios", f))
                speech = librosa.resample(speech.squeeze().numpy(), orig_sr=sample_rate, target_sr=16000)

                duration = speech.shape[-1] / 16000
                # print(duration)
                # s()
                for w in asr[f][f"{basic}_offsets"]:
                    time.append([w["start_time"] + offset, w["end_time"] + offset])
                    if len(time) > 2:
                        if not (time[-1][0] >= time[-2][1]):
                            print(time[-2:])
                            # s()
                transcription += asr[f]["asr"]
                transcription += " "

                offset += duration

            question_times.append(time)
            question_transcriptions.append(transcription)

df["context_asr"] = content_transcriptions
df["context_times"] = content_times
df["question_asr"] = question_transcriptions
df["question_times"] = question_times
df.to_parquet(f"./asr_results/{model_name}/test_phone.parquet")
