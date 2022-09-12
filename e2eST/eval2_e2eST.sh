n=4502
while true
do
    FILE=facebook/wav2vec2-large-lv60_google/byt5-base_SpeechMixEEDT5_wav2vec2-large-lv60_byt5-base_textdecoderonly_bs64/checkpoint-${n}
    if [ -f "${FILE}/pytorch_model.bin" ]; then
        echo "running eval ${n}"
        python3 eval_hf_fast.py --speech_model_config facebook/wav2vec2-large-lv60 \
        --nlp_model_config google/byt5-base \
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

