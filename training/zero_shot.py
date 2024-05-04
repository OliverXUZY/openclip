import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from .precision import get_autocast
import os

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args, drop_block_masks = None):
    """
    Args:
        drop_block_masks (bool tensor, [n, ] ): masks for residual connections. n is num of knobs
    """
    drop_block_masks_ori = drop_block_masks.clone() if drop_block_masks is not None else None

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)
    # counter = 0
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            # print(target)
            # counter += 1
            # if counter >= 10:
            #     break

            images = images.to(device=args.device, dtype=input_dtype) # torch.float32 [bs, c, h, w] [bs, 3, 224, 224]
            target = target.to(args.device) # [bs, ]
            if drop_block_masks_ori is not None:
                drop_block_masks = drop_block_masks_ori.repeat(images.shape[0], 1)  #  [bs, n]
            else:
                drop_block_masks = None

            with autocast():
                # predict
                output = model(image=images, drop_block_masks=drop_block_masks)
                image_features = output['image_features'] if isinstance(output, dict) else output[0] # [bs, dim] [64, 512]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results



def zero_shot_eval_macs(model, data, epoch, args, tokenizer=None, drop_block_masks = None):
    """
    Args:
        drop_block_masks (bool tensor, [n, ] ): masks for residual connections. n is num of knobs
    """
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    
    # Path to the saved tensor file
    file_path = 'imagenet_val_classifier_tensor.pt'
    if os.path.exists(file_path):
        print(f"Load the tensor if the file exists in {file_path}")
        classifier = torch.load(file_path)
    else:
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            ) # tensor [embd_dim, num_class] [512, 1000]
        
        # Save the new classifier tensor to the file
        print(f"Save the new classifier tensor to the file in {file_path}")
        torch.save(classifier, file_path)

    
    # print("temp zhuoyan !!!")
    # classifier = torch.randn([512,1000], device="cuda")
    # print(classifier.shape, type(classifier), classifier.dtype, classifier.device)
    # assert False

    logging.info('Using classifier')
    results = {}
    model_macs = model.visual.get_macs().to("cuda")
    # print("==== drop_block_masks====: ", drop_block_masks.shape, drop_block_masks)
    # print("model_macs: ", model_macs)
    returned_macs = (drop_block_masks * model_macs[1:]).sum(dim=-1) + model_macs[0]
    # print("===========", returned_macs.item())
    # assert False
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args, drop_block_masks = drop_block_masks)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
        results['macs'] = returned_macs.item()
    else:
        assert False, "not implement yet for imagenet-v2"

    logging.info('Finished zero-shot imagenet.')

    return results
