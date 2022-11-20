from transformers import BertModel, BertConfig, T5Model
md = BertModel.from_pretrained("bert-base-uncased")
print(md)

cfg = BertConfig.from_pretrained("bert-base-uncased")
print(cfg)

cfg.h


md2 = T5Model.from_pretrained("google/byt5-small")
print(md2)


