python train_vit.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --imagenet-train=data/imagenet/train \
    --imagenet-val=data/imagenet/val \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    -g 1 \
    # --report-to wandb \
    # --wandb-project-name openclip_ft \