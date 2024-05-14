python eval_vit.py \
    --imagenet-val data/imagenet/val \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    -g 0 \
    --resume ./logs/2024_05_13-20_57_49-model_ViT-B-32-quickgelu-lr_0.001-b_64-j_8-p_amp/checkpoints/epoch_last.pt \
