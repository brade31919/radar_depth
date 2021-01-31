import os
import time
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision.utils import make_grid
cudnn.benchmark = True

from model.utils import Result as Result_point
from model.utils import AverageMeter as AverageMeter_point
# Models and modified models from sparse to dense
from model.models import (
    ResNet, 
    ResNet2, 
    ResNet_latefusion
)
# The multi-stage model variants
from model.multistage_model import ResNet_multistage
from evaluation.metrics import AverageMeter, Result
from tensorboardX import SummaryWriter
import utils
from evaluation.criteria_new import (
    MaskedCrossEntropyLoss, 
    SmoothnessLoss,
    MaskedMSELoss,
    MaskedL1Loss
)
from dataset.nuscenes_dataset_torch_new import nuscenes_dataset_torch
import torch.utils.data.dataloader as torch_loader

args = utils.parse_command()

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']

best_result = Result()
best_result.set_to_worst()

multistage_group = ['resnet18_multistage', 'resnet18_multistage_uncertainty', 'resnet18_multistage_uncertainty_fixs']
uncertainty_group = ['resnet18_multistage_uncertainty', 'resnet18_multistage_uncertainty_fixs']
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Define the customized collate_fn
def customized_collate(batch):
    list_keys = ["daynight_info"]
    batch_keys = ['rgb', 'lidar_depth', 'radar_depth', 'inputs', \
                  'labels', 'index_map']
    outputs = {}
    for key in batch_keys:                                                       
        outputs[key] = torch_loader.default_collate([b[key] for b in batch])
    for key in list_keys:                                                        
        outputs[key] = [b[key] for b in batch] 

    return outputs


# Create dataloader given input arguments
def create_data_loaders(args):
    # Data loading code
    print("[Info] Creating data loaders ...")
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf

    if args.data == "nuscenes":
        if not args.evaluate:
            train_dataset = nuscenes_dataset_torch(
                "train",
                transform_mode="sparse-to-dense",
                modality=args.modality,
                sparsifier=args.sparsifier,
                num_samples=args.num_samples,
                max_depth=max_depth
            )
        if args.validation:
            val_dataset = nuscenes_dataset_torch(
                "val",
                transform_mode="sparse-to-dense",
                modality=args.modality,
                sparsifier=args.sparsifier,
                num_samples=args.num_samples,
                max_depth=max_depth
            )
    else:
        raise RuntimeError('[Error] Dataset not found. The dataset must be nuscenes')

    if args.validation:
        # Always use batch_size=1 in validation
        val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=1, num_workers=4, shuffle=False, pin_memory=True,
            collate_fn=customized_collate
        )

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id)
        ) # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    if args.validation:
        return train_loader, val_loader
    else:
        return train_loader


# Create model given input arguments and output size
def create_model(args, output_size):
    print(f"[Info] Creating Model ({args.arch}-{args.decoder}) ...")
    in_channels = len(args.modality)
    if args.arch == 'resnet50':
        model = ResNet(layers=50, decoder=args.decoder, output_size=output_size,
                       in_channels=in_channels, pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = ResNet(layers=18, decoder=args.decoder, output_size=output_size,
                       in_channels=in_channels, pretrained=args.pretrained)
    elif args.arch == "resnet34":
        model = ResNet(layers=34, decoder=args.decoder, output_size=output_size,
                       in_channels=in_channels, pretrained=args.pretrained)
    elif args.arch == 'resnet18_new':
        model = ResNet2(layers=18, decoder=args.decoder, output_size=output_size,
                        in_channels=in_channels, pretrained=args.pretrained)
    elif args.arch == "resnet18_latefusion":
        model = ResNet_latefusion(layers=18, decoder=args.decoder, output_size=output_size,
                                  in_channels=in_channels, pretrained=args.pretrained)
    elif args.arch == "resnet18_multistage":
        model = ResNet_multistage(layers=18, decoder=args.decoder, output_size=output_size,
                                  pretrained=args.pretrained)
    # If uncertainty model, we need to add weighting parameters to the model
    elif args.arch == "resnet18_multistage_uncertainty":
        model = ResNet_multistage(layers=18, decoder=args.decoder, output_size=output_size,
                                  pretrained=args.pretrained)
        # Get loss weights
        w_stage1 = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)
        w_stage2 = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)
        w_smooth = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=True)

        # Register the parameters to the model
        model.register_parameter("w_stage1", w_stage1)
        model.register_parameter("w_stage2", w_stage2)
        model.register_parameter("w_smooth", w_smooth)

        loss_weights = {
            "w_stage1": w_stage1,
            "w_stage2": w_stage2,
            "w_smooth": w_smooth
        }

        return model, loss_weights
    
    # If the fixs model, we have deterministic weights for smoothness loss
    elif args.arch == "resnet18_multistage_uncertainty_fixs":
        model = ResNet_multistage(layers=18, decoder=args.decoder, output_size=output_size,
                                  pretrained=args.pretrained)
        # Get loss weights
        w_stage1 = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)
        w_stage2 = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)
        w_smooth = 0.1

        # Register the parameters to the model
        model.register_parameter("w_stage1", w_stage1)
        model.register_parameter("w_stage2", w_stage2)

        loss_weights = {
            "w_stage1": w_stage1,
            "w_stage2": w_stage2,
            "w_smooth": w_smooth
        }

        return model, loss_weights

    else:
        raise ValueError("[Error] Unknown model!!")
    print("[Info] model created.")

    return model


def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        f"[Error] Can't find the specified checkpoint at '{args.evaluate}'"
        print(f"[Info] loading the model '{args.evaluate}'")
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        print(args)
        train_loader, val_loader = create_data_loaders(args)
        model_weights = checkpoint['model_state_dict']
        # Create model
        if args.arch == "resnet18_multistage_uncertainty" or \
           args.arch == "resnet18_multistage_uncertainty_fixs":
            model, loss_weights = create_model(args, output_size=train_loader.dataset.output_size)
        else:
            model = create_model(args, output_size=train_loader.dataset.output_size)
            loss_weights = None
        model.load_state_dict(model_weights, strict=False)
        model = model.cuda()
        print(f"[Info] Loaded best model (epoch {checkpoint['epoch']})")
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            f"[Info] No checkpoint found at '{chkpt_path}'"
        print(f"=> loading checkpoint '{chkpt_path}'")
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        print(args)
        start_epoch = checkpoint['epoch'] + 1
        try:
            best_result = checkpoint['best_result']
        except:
            best_result.set_to_worst()

        # Create dataloader first
        args.validation = True
        args.workers = 8

        if (args.data == "nuscenes") and (args.modality == "rgbd") and (args.sparsifier == "uar"):
            args.sparsifier = None
        # Create dataloader
        if args.validation:
            train_loader, val_loader = create_data_loaders(args)
        else:
            train_loader = create_data_loaders(args)
        # Load from model's state dict instead
        model_weights = checkpoint['model_state_dict']
        # Create model
        if args.arch == "resnet18_multistage_uncertainty" or \
           args.arch == "resnet18_multistage_uncertainty_fixs":
            model, loss_weights = create_model(args, output_size=train_loader.dataset.output_size)
        else:
            model = create_model(args, output_size=train_loader.dataset.output_size)
            loss_weights = None
        model.load_state_dict(model_weights, strict=False)
        model = model.cuda()

        # Create optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            args.lr,
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        args.resume = True
    # Create new model
    else:
        print(args)
        # Create dataloader
        if args.validation:
            train_loader, val_loader = create_data_loaders(args)
        else:
            train_loader = create_data_loaders(args)

        # Create model
        if args.arch == "resnet18_multistage_uncertainty" or \
           args.arch == "resnet18_multistage_uncertainty_fixs":
            model, loss_weights = create_model(args, output_size=train_loader.dataset.output_size)
        else:
            model = create_model(args, output_size=train_loader.dataset.output_size)
            loss_weights = None

        # Create optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            args.lr,
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
        model = model.cuda()

    # Define loss function (criterion) and optimizer
    criterion = {}
    if args.criterion == 'l2':
        criterion["depth"] = MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion["depth"] = MaskedL1Loss().cuda()
    else:
        raise ValueError("[Error] Unknown criterion...")
    
    # Add smoothness loss to the criterion
    if args.arch == "resnet18_multistage_uncertainty" or \
       args.arch == "resnet18_multistage_uncertainty_fixs":
        criterion["smooth"] = SmoothnessLoss().cuda()

    # Create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # Create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Create summary writer
    log_path = os.path.join(output_directory, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        # Adjust the learning rate
        utils.adjust_learning_rate(optimizer, epoch, args.lr)

        # Record the learning rate summary
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            logger.add_scalar('Lr/lr_' + str(i), old_lr, epoch)

        # Perform training (train for one epoch)
        train(train_loader, model, criterion, optimizer, epoch, loss_weights, logger=logger)

        # Perform evaluation
        if args.validation:
            result, img_merge = validate(val_loader, model, epoch, logger=logger)

            is_best = result.rmse < best_result.rmse
            if is_best:
                best_result = result
                with open(best_txt, 'w') as txtfile:
                    txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
                if img_merge is not None:
                    img_filename = output_directory + '/comparison_best.png'
                    utils.save_image(img_merge, img_filename)

        # Save different things in different mode
        if args.validation:
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'best_result': best_result,
                'optimizer_state_dict' : optimizer.state_dict(),
            }, is_best, epoch, output_directory)
        else:
            utils.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, False, epoch, output_directory)


def train(train_loader, model, criterion, optimizer, epoch, loss_weights=None, logger=None):
    # pdb.set_trace()
    average_meter = AverageMeter()
    if args.arch in multistage_group:
        average_meter_stage1 = AverageMeter()

    model.train() # switch to train mode
    end = time.time()

    # Record number of batches
    batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        ############ Fetch input data ################
        # Add compatibility for nuscenes
        if args.data != "nuscenes":
            inputs, target = data[0].cuda(), data[1].cuda()
        else:
            inputs, target = data["inputs"].cuda(), data["labels"].cuda()
        
        torch.cuda.synchronize()
        data_time = time.time() - end

        # Training step
        end = time.time()
        if args.arch == "resnet18_multistage_uncertainty":
            pred_ = model(inputs)
            pred1 = pred_["stage1"]
            pred = pred_["stage2"]
            depth_loss1 = criterion["depth"](pred1, target)
            depth_loss2 = criterion["depth"](pred, target)
            smooth_loss = criterion["smooth"](pred1, input)
            weight_loss = loss_weights["w_stage1"] + loss_weights["w_stage2"] + loss_weights["w_smooth"]

            # Weighted sum to total loss
            loss = torch.exp(-loss_weights["w_stage1"]) * depth_loss1 + \
                   torch.exp(-loss_weights["w_stage2"]) * depth_loss2 + \
                   torch.exp(-loss_weights["w_smooth"]) * smooth_loss + \
                   weight_loss

        elif args.arch == "resnet18_multistage_uncertainty_fixs":
            pred_ = model(inputs)
            pred1 = pred_["stage1"]
            pred = pred_["stage2"]
            depth_loss1 = criterion["depth"](pred1, target)
            depth_loss2 = criterion["depth"](pred, target)
            smooth_loss = criterion["smooth"](pred1, inputs)
            weight_loss = loss_weights["w_stage1"] + loss_weights["w_stage2"]

            # Weighted sum to total loss
            stage1_weighted_loss = torch.exp(-loss_weights["w_stage1"]) * (depth_loss1 + (loss_weights["w_smooth"] * smooth_loss))
            stage2_weighted_loss = torch.exp(-loss_weights["w_stage2"]) * depth_loss2
            loss = stage1_weighted_loss + stage2_weighted_loss + \
                   weight_loss

        elif args.arch in multistage_group:
            pred_ = model(inputs)
            pred1 = pred_["stage1"]
            pred = pred_["stage2"]
            depth_loss1 = criterion["depth"](pred1, target)
            depth_loss2 = criterion["depth"](pred, target)
            loss = depth_loss1 + depth_loss2

        else:
            pred = model(inputs)
            loss = criterion["depth"](pred, target)

        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # [Depth] Measure error and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, inputs.size(0))

        # [Depth] Measure stage1 error
        if args.arch in multistage_group:
            result_stage1 = Result()
            result_stage1.evaluate(pred1.data, target.data)
            average_meter_stage1.update(result_stage1, gpu_time, data_time, inputs.size(0))

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f})'.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))

        if ((i + 1) % 100 == 0) and (logger is not None):
            current_step = epoch * batch_num + i
            # Add scalar summaries
            logger.add_scalar('Train_loss/Loss', loss.item(), current_step)
            record_scalar_summary(result, average_meter, current_step, logger, "Train")

            # Further add some scalar summaries
            if args.arch == "resnet18_multistage_uncertainty":
                # Add weight summaries
                logger.add_scalar("Train_weights/w_stage1", torch.exp(-loss_weights["w_stage1"]).item(), current_step)
                logger.add_scalar("Train_weights/w_stage2", torch.exp(-loss_weights["w_stage2"]).item(), current_step)
                logger.add_scalar("Train_weights/w_smooth", torch.exp(-loss_weights["w_smooth"]).item(), current_step)

                # Add loss summary
                logger.add_scalar("Train_loss/Smoothness_loss", smooth_loss.item(), current_step)
                logger.add_scalar("Train_loss/Weight_loss", weight_loss.item(), current_step)
            
            # Some scalar summaries for the uncertainty fixs model
            if args.arch == "resnet18_multistage_uncertainty_fixs":
                # Add weight summaries
                logger.add_scalar("Train_weights/w_stage1", torch.exp(-loss_weights["w_stage1"]).item(), current_step)
                logger.add_scalar("Train_weights/w_stage2", torch.exp(-loss_weights["w_stage2"]).item(), current_step)
                
                # Add loss summary
                logger.add_scalar("Train_loss/Smoothness_loss", smooth_loss.item(), current_step)
                logger.add_scalar("Train_loss/Weight_loss", weight_loss.item(), current_step)

                # Add weighted loss
                logger.add_scalar("Train_loss_weighted/stage1", stage1_weighted_loss.item(), current_step)
                logger.add_scalar("Train_loss_weighted/stage2", stage2_weighted_loss.item(), current_step)
            
            if args.arch in multistage_group:
                logger.add_scalar('Train_loss/Depth_loss1', depth_loss1.item(), current_step)
                logger.add_scalar('Train_loss/Depth_loss2', depth_loss2.item(), current_step)
                # Record error summaries for stage1
                record_scalar_summary(result_stage1, average_meter_stage1, current_step, logger, "Train_stage1")

            # Add system info
            logger.add_scalar('System/gpu_time', average_meter.average().gpu_time, current_step)
            logger.add_scalar('System/data_time', average_meter.average().data_time, current_step)

            # Add some image summary
            if args.modality == "rgb":
                input_images = inputs.cpu()
            else:
                input_images = inputs[:, 0:3, :, :].cpu()
                input_depth = torch.unsqueeze(inputs[:, 3, :, :], dim=1).cpu()
            rgb_grid = make_grid(input_images[0:6, :, :, :], nrow=3, normalize=False),
            target_grid = make_grid(target.cpu()[0:6, :, :, :], nrow=3, normalize=True, range=(0, 80))
            pred_grid = make_grid(pred.cpu()[0:6, :, :, :], nrow=3, normalize=True, range=(0,80))
            logger.add_image('Train/RGB', rgb_grid[0].data.numpy())
            logger.add_image('Train/Depth_gt', target_grid.data.numpy())
            logger.add_image('Train/Depth_pred', pred_grid.data.numpy())

            # Also record depth predictions from stage1
            if args.arch in multistage_group:
                pred_grid1 = make_grid(pred1.cpu()[0:6, :, :, :], nrow=3, normalize=True, range=(0,80))
                logger.add_image('Train/Depth_pred1', pred_grid1.data.numpy())
            if args.modality == "rgbd":
                depth_grid = make_grid(input_depth[0:6, :, :, :], nrow=3, normalize=True, range=(0, 80))
                logger.add_image('Train/Depth_input', depth_grid.data.numpy())

        end = time.time()

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True, logger=None):
    average_meter = AverageMeter()
    if args.arch in multistage_group:
        average_meter_stage1 = AverageMeter()
    
    # Include daynight info and rain condition
    avg_meter_day = AverageMeter()
    avg_meter_night = AverageMeter()

    # day, night, sun, rain combinations
    avg_meter_day_sun = AverageMeter()
    avg_meter_day_rain = AverageMeter()
    avg_meter_night_sun = AverageMeter()
    avg_meter_night_rain = AverageMeter()

    # sun and rain
    avg_meter_sun = AverageMeter()
    avg_meter_rain = AverageMeter()

    model.eval() # switch to evaluate mode
    end = time.time()

    # Save something to draw??
    if logger is None:
        import h5py
        output_path = os.path.join(output_directory, "results.h5")
        h5_writer = h5py.File(output_path, "w", libver="latest", swmr=True)

    for i, data in enumerate(val_loader):
        # Add compatibility for nuscenes
        if args.data != "nuscenes":
            inputs, target = data[0].cuda(), data[1].cuda()
        else:
            inputs, target = data["inputs"].cuda(), data["labels"].cuda()
            
        torch.cuda.synchronize()
        data_time = time.time() - end

        # Compute output
        end = time.time()
        with torch.no_grad():
            if args.arch in multistage_group:
                pred_ = model(inputs)
                pred1 = pred_["stage1"]
                pred = pred_["stage2"]
            else:
                pred = model(inputs)
                pred_ = None

        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # Record for qualitative results
        if (logger is None) and (i % 5 == 0):
            pred_np = {}
            if pred_ is None:
                pred_np = pred.cpu().numpy()
            else:
                for key in pred_.keys():
                    pred_np[key] = pred_[key][0, ...].cpu().numpy()
            res = {
                "inputs": data["inputs"][0, ...].cpu().numpy(),
                "lidar_depth": data["lidar_depth"][0, ...].cpu().numpy(),
                "radar_depth": data["radar_depth"][0, ...].cpu().numpy(),
                "pred": pred_np
            }
            file_key = "%05d"%(i)
            f_group = h5_writer.create_group(file_key)
            # Store data
            for key, output_data in res.items():
                if isinstance(output_data, dict):
                    for key, data_ in output_data.items():
                        if key in res.keys():
                            key = key + "*"
                        f_group.create_dataset(key, data=data_, compression="gzip")
                elif output_data is None:
                    pass
                else:    
                    f_group.create_dataset(key, data=output_data, compression="gzip")

        # Measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, inputs.size(0))
        if args.arch in multistage_group:
            result_stage1 = Result()
            result_stage1.evaluate(pred1.data, target.data)
            average_meter_stage1.update(result_stage1, gpu_time, data_time, inputs.size(0))
        end = time.time()
        
        # Record the day, night, rain info
        assert inputs.size(0) == 1
        daynight_info = data["daynight_info"][0]
        if ("day" in daynight_info) and ("rain" in daynight_info):
            avg_meter_day_rain.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_day.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_rain.update(result, gpu_time, data_time, inputs.size(0))
        elif "day" in daynight_info:
            avg_meter_day_sun.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_day.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_sun.update(result, gpu_time, data_time, inputs.size(0))

        if ("night" in daynight_info) and ("rain" in daynight_info):
            avg_meter_night_rain.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_night.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_rain.update(result, gpu_time, data_time, inputs.size(0))
        elif "night" in daynight_info:
            avg_meter_night_sun.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_night.update(result, gpu_time, data_time, inputs.size(0))
            avg_meter_sun.update(result, gpu_time, data_time, inputs.size(0))

        
        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = inputs
            elif args.modality == 'rgbd':
                rgb = inputs[:,:3,:,:]
                depth = inputs[:,3:,:,:]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    # Save the result to pkl file
    if logger is None:
        h5_writer.close()
    avg = average_meter.average()
    if args.arch in multistage_group:
        avg_stage1 = average_meter_stage1.average()
        if logger is not None:
            record_test_scalar_summary(avg_stage1, epoch, logger, "Test_stage1")

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'Rel={average.absrel:.3f}\n'
          'Log10={average.lg10:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'Delta2={average.delta2:.3f}\n'
          'Delta3={average.delta3:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if logger is not None:
        # Record summaries
        record_test_scalar_summary(avg, epoch, logger, "Test")

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

    return avg, img_merge


def record_scalar_summary(result, avg, current_step, logger, prefix="Train"):
    # Add scalar summaries
    logger.add_scalar(prefix + '_error/RMSE', result.rmse, current_step)
    logger.add_scalar(prefix + '_error/rel', result.absrel, current_step)
    logger.add_scalar(prefix + '_error/mae', result.mae, current_step)
    logger.add_scalar(prefix + '_delta/Delta1', result.delta1, current_step)
    logger.add_scalar(prefix + '_delta/Delta2', result.delta2, current_step)
    logger.add_scalar(prefix + '_delta/Delta3', result.delta3, current_step)

    # Add smoothed summaries
    average = avg.average()
    logger.add_scalar(prefix + '_error_smoothed/RMSE', average.rmse, current_step)
    logger.add_scalar(prefix + '_error_smoothed/rml', average.absrel, current_step)
    logger.add_scalar(prefix + '_error_smoothed/mae', average.mae, current_step)
    logger.add_scalar(prefix + '_delta_smoothed/Delta1', average.delta1, current_step)
    logger.add_scalar(prefix + '_delta_smoothed/Delta2', average.delta2, current_step)
    logger.add_scalar(prefix + '_delta_smoothed/Delta3', average.delta3, current_step)


def record_test_scalar_summary(avg, epoch, logger, prefix="Test"):
    logger.add_scalar(prefix + '/rmse', avg.rmse, epoch)
    logger.add_scalar(prefix + '/Rel', avg.absrel, epoch)
    logger.add_scalar(prefix + '/log10', avg.lg10, epoch)
    logger.add_scalar(prefix + '/Delta1', avg.delta1, epoch)
    logger.add_scalar(prefix + '/Delta2', avg.delta2, epoch)
    logger.add_scalar(prefix + '/Delta3', avg.delta3, epoch)

def display_results(avg_meter):
    avg = avg_meter.average()
    print("RMSE:", avg.rmse)
    print("MAE:", avg.mae)
    print("REL", avg.absrel)
    print("log10", avg.lg10)
    print("delta1", avg.delta1)
    print("delta2", avg.delta2)
    print("delta3", avg.delta3)


if __name__ == '__main__':
    main()