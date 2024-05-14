import torch
import math
import time
import logging

from .train import AverageMeter, backward, unwrap_model
from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from .distributed import is_master
from .zero_shot import zero_shot_eval, zero_shot_eval_macs
from .precision import get_autocast
from training.zero_shot import accuracy
from src.utils import Timer, time_str
try:
    import wandb
except ImportError:
    wandb = None

def train_one_epoch_vit(
        model, 
        data, 
        loss, 
        epoch, 
        optimizer,
        scaler, 
        scheduler, 
        dist_model, 
        args, 
        tb_writer=None, 
        ada_scheduler = None,
        text_classifier = None
        ):
    """
        Args:
            ada_scheduler (ada_Scheduler): the scheduler output policy
        """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    ada_scheduler.train()
    if args.distill:
        dist_model.eval()

    data['imagenet-train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['imagenet-train'].dataloader
    # print("dataloader.num_batches: ", dataloader.num_batches, "args.accum_freq", args.accum_freq)
    # assert False
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, target = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        target = target.to(args.device) # [bs, ]

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        ### construct latency
        upper = 1.0
        # Create a new local generator
        local_rng = torch.Generator()
        local_rng.manual_seed(42)
        latency, _ = torch.rand(target.shape[0], generator=local_rng).sort()
        latency = 0.1073 + (latency * (upper - 0.1073))
        latency = latency.to("cuda")

        if args.accum_freq == 1:
            with autocast():
                # drop_block_masks = torch.tensor([True, False, True, True, True, False, True, True, True, False, True]).to("cuda").unsqueeze(0).repeat(target.shape[0], 1)
                # model_out = model(image=images, drop_block_masks=drop_block_masks)
                model_out = model(image=images, drop_block_masks=None, latency = latency, ada_scheduler_forward = ada_scheduler.forward)
                logit_scale = model_out["logit_scale"]

                image_features = model_out['image_features'] if isinstance(model_out, dict) else model_out[0] # [bs, dim] [64, 512]
                returned_macs = model_out['returned_macs'] if isinstance(model_out, dict) else model_out[-1] # [bs,] [64]
                logits = 100. * image_features @ text_classifier

                
                losses = loss(logits, target, macs = returned_macs, latency = latency)

                total_loss = losses['ada_loss']
                losses["loss"] = total_loss
                # print("total_loss: ", total_loss)
                # assert False
            
            

            backward(total_loss, scaler)
        else:
            assert False, "Not implement yet"

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def eval_vit(
        model, 
        data, 
        dist_model, 
        args, 
        tb_writer=None, 
        ada_scheduler = None,
        text_classifier = None,
        num_latency = 128,
    ):
    upper = 1.0
    local_rng = torch.Generator()
    local_rng.manual_seed(42)
    latency, _ = torch.rand(num_latency, generator=local_rng).sort()

    # Create a tensor for the value 1 with the same dtype and device as 'latency'
    value_to_append = torch.tensor([1], dtype=latency.dtype, device=latency.device)
    # Append the value to the 'latency' tensor
    latency = torch.cat((latency, value_to_append))

    latency = 0.1073 + (latency * (upper - 0.1073))
    latency = latency.to("cuda")
    timer = Timer()

    top1s = []
    top5s = []
    macs_diff = []
    macss = []
    for idx, laten in enumerate(latency):
        top1, top5, total_macs, total_macs_diff = eval_vit_one_latency(
            model, 
            data, 
            dist_model, 
            args, 
            tb_writer, 
            ada_scheduler,
            text_classifier,
            latency = laten
        )
        time_elapsed = timer.end()
        print(f"latency {idx} | {latency.shape[0]} \
                time elapsed: {time_str(time_elapsed)} | {time_str(time_elapsed/(idx + 1)*latency.shape[0])}")

        top1s.append(top1)
        top5s.append(top5)
        macs_diff.append(total_macs_diff)
        macss.append(total_macs)
    
    ## save and print
    metric_list = {
        "accs": top1s,
        "top5s": top5s,
        "macs_diff": macs_diff,
        "latencys": latency,
        "macs": macss
    }
    log_str = 'Results:\n'
    # Column names
    headers = list(metric_list.keys())
    log_str += '{:15s}\t{:15s}\t{:15s}\t{:15s}\n'.format(*headers)

    # Iterate through each metric list by index since they're of the same length
    for i in range(len(next(iter(metric_list.values())))):
        # Fetch values from each list by index and format
        row = [metric_list[metric][i] for metric in metric_list]
        log_str += '{:<15.2f}\t{:<15.2f}\t{:<15.2f}\t{:<15.2f}\n'.format(*row)
    print("log_str: ", log_str)

    results = []
    for top1, top5, mac_diff, laten, macs in zip(top1s, top5s, macs_diff, latency, macss):
        result = {}
        result['imagenet-zeroshot-val-top1'] = top1
        result['imagenet-zeroshot-val-top5'] = top5
        result['macs_diff'] = mac_diff
        result['latency'] = laten  # in zero-shot, calculated_macs is a vector with bs, all items are the same values
        result['macs'] = macs
        results.append(result)
    
    return results


def eval_vit_one_latency(
        model, 
        data, 
        dist_model, 
        args, 
        tb_writer=None, 
        ada_scheduler = None,
        text_classifier = None,
        latency = None
        ):
    """
    Args:
        ada_scheduler (ada_Scheduler): the scheduler output policy
        latency (float tensor, (bs,)): latency for each input.
    """

    model.eval()
    ada_scheduler.eval()
    latency_ori = latency.clone()
    
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if args.distill:
        dist_model.eval()

    dataloader = data['imagenet-val'].dataloader
    # print("dataloader.num_batches: ", dataloader.num_batches, "args.accum_freq", args.accum_freq)
    # assert False
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    data_time_m = AverageMeter()
    end = time.time()
    total_macs_diff = 0
    total_macs  = 0

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, batch in enumerate(dataloader):
            i_accum = i // args.accum_freq

            images, target = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            target = target.to(args.device) # [bs, ]

            data_time_m.update(time.time() - end)

            ### construct latency
            latency = latency_ori.repeat(target.shape[0]).cuda()

            with autocast():
                model_out = model(image=images, drop_block_masks=None, latency = latency, ada_scheduler_forward = ada_scheduler.forward)
                logit_scale = model_out["logit_scale"]

                image_features = model_out['image_features'] if isinstance(model_out, dict) else model_out[0] # [bs, dim] [64, 512]
                returned_macs = model_out['returned_macs'] if isinstance(model_out, dict) else model_out[-1] # [bs,] [64]
                logits = 100. * image_features @ text_classifier
            
            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)
            diff = returned_macs - latency
            total_macs_diff += diff[diff > 0].sum().item()
            total_macs += returned_macs.sum().item()

    top1 = (top1 / n)
    top5 = (top5 / n)
    total_macs_diff = (total_macs_diff / n)
    total_macs = (total_macs / n)
    
    return top1, top5, total_macs, total_macs_diff