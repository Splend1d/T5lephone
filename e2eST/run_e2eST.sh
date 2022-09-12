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

