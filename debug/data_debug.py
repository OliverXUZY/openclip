import sys
sys.path.insert(0, "/home/user/Documents/projects/openclip/")
sys.path.insert(0, "/home/zhuoyan/vision/openclip")
from tqdm import tqdm

from training.data import get_data
from training.params import parse_args

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.distributed import is_master, init_distributed_device, broadcast_object


def main(args):
    # fully initialize distributed device environment
    device = init_distributed_device(args)
    print("device: ", device)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )

    start_epoch = 0

    # print("preprocess_train: ", preprocess_train)
    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    # print("tolenizer: ", tokenizer)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'
    print("data: ", data.keys())

    print("=========================== val =================================")
    image_val = data['imagenet-val']
    val_loader = image_val.dataloader
    val_set = val_loader.dataset
    print("len val dataset: ", len(val_set))

    eg = val_set[0]
    print(len(eg), type(eg))
    print(eg[0].shape, eg[1])

    eg = next(iter(val_loader))
    print(len(eg), type(eg))
    print(eg[0].shape, eg[1])

    # for images, target in tqdm(val_loader, unit_scale=args.batch_size):
    #     print(target)

    print("=========================== train =================================")
    image_train = data['imagenet-train']
    train_loader = image_train.dataloader
    train_set = train_loader.dataset
    print("len train dataset: ", len(train_set))

    eg = train_set[0]
    print(len(eg), type(eg))
    print(eg[0].shape, eg[1])

    eg = next(iter(train_loader))
    print(len(eg), type(eg))
    print(eg[0].shape, eg[1])



if __name__ == "__main__":
    # print(sys.argv)
    # print(sys.argv[1:])
    args = sys.argv[1:]
    # print(args)
    args = parse_args(args)

    main(args)