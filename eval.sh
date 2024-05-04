# python -m training.main \
#     --imagenet-val data/imagenet/val \
#     --model ViT-B-32-quickgelu \
#     --pretrained laion400m_e32


python main.py \
    --imagenet-val data/imagenet/val \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32 \
    -g 0