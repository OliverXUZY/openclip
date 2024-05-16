python eval_vit.py \
    --imagenet-val data/imagenet/val \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    -g 0 \
    --resume ./logs/eps3_1gpu_lora/checkpoints/epoch_last.pt \
    --num_latency 3 \
