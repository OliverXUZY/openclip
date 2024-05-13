python debug/data_debug.py \
    --imagenet-train data/imagenet/train \
    --imagenet-val data/imagenet/val \
    --model ViT-B-32-quickgelu \
    --pretrained laion400m_e32

# python debug/model_forward.py \
#     --imagenet-val data/imagenet/val \
#     --model ViT-B-32-quickgelu \
#     --pretrained laion400m_e32

# python debug/block_forward.py 

# python debug/optimi.py \
#     --imagenet-val data/imagenet/val \
#     --model ViT-B-32-quickgelu \
#     --pretrained laion400m_e32