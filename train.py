import argparse
import logging
import os
import pprint
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from openpyxl import Workbook

from util.classes import CLASSES
from util.dataset_acdc import ACDCDataset
from util.dataset_rgb import RGBDataset
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.utils2 import compute_entropy_uncertainty_torch, UDAT  # Uncertainty Distribution Adaptive Thresholding
from util.utils3 import (get_model, apply_mask_cutmix, calculate_batch_metrics, update_metrics_lists,
                         calculate_metrics, apply_mixup, apply_mask_mixup) # model and util
from util.utils4 import exchange_patches # Bidirectional Semantic Consistency-aware Exchange

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Semi-URF')
parser.add_argument('--data_mode', type=str, default='L', choices=['L', 'RGB'],
                    help='Data mode: L (ACDC, promise) or rgb (ISIC)')
parser.add_argument('--dataset', type=str, default='acdc', help='dataset name: acdc promise isic')
parser.add_argument('--data_root', type=str, default='data/ACDC', help='dataset path')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='epoch')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--crop_size', type=int, default=256, help='image_size')
parser.add_argument('--nclass', type=int, default=4, help='number class')
parser.add_argument('--image_size', type=int, default=256, help='image_size')
parser.add_argument('--labeled_data_list', default='splits/acdc/7/labeled.txt', type=str)
parser.add_argument('--unlabeled_data_list', default='splits/acdc/7/unlabeled.txt', type=str)
parser.add_argument('--save-path', type=str, default='semi-URF')
parser.add_argument('--optimizer', default='optimizer_Adam', type=str,
                    help='optimizer_SGD or optimizer_Adam')

# Record evaluation metrics
(medpy_dice_list, hd_list, hd95_list, iou_list, precision_list,
 recall_list, specificity_list, asd_list) = [], [], [], [], [], [], [], []
def main():
    args = parser.parse_args()
    wb = Workbook()
    # Record evaluation metrics
    ws1 = wb.create_sheet('estimation', 1)
    ws1.append(["medpy_dice", "hd", "hd95", "iou", "precision", "recall", "specificity", "asd"])
    # Modify model and recording save location
    current_experiment_time = datetime.now().strftime('%Y%m%d%H%M%S').replace(":", "")
    snapshot_path = "experiment/{}/{}".format(
        current_experiment_time, args.save_path)
    writer = SummaryWriter(snapshot_path)
    os.makedirs(snapshot_path, exist_ok=True)
    # Log recording module
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = vars(args)
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(snapshot_path + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    cudnn.enabled = True
    cudnn.benchmark = True
    # Model definition
    model = get_model(args)
    model.cuda()
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    # Optimizer
    if args.optimizer == 'optimizer_SGD':
        optimizer = SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'optimizer_Adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=args.nclass)  # Input requirement: [B, C, H, W] and [B, 1, H, W]
    if args.data_mode == 'L':
        dataset_class = ACDCDataset
    elif args.data_mode == 'RGB':
        dataset_class = RGBDataset
    trainset_u = dataset_class(args.dataset, args.data_root, 'train_u', args.crop_size, args.unlabeled_data_list)
    trainset_l = dataset_class(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_data_list,
                               nsample=len(trainset_u.ids))
    valset = dataset_class(args.dataset, args.data_root, 'val')
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size, pin_memory=True, num_workers=1, drop_last=True)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=args.batch_size, pin_memory=True, num_workers=1,
                                   drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    total_iters = len(trainloader_u) * args.epochs
    previous_best = 0.0
    epoch = -1
    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    for epoch in range(epoch + 1, args.epochs):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))
        total_loss = AverageMeter()
        total_mask_ratio = AverageMeter()
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            # img_x, mask_x: labeled data, mask is label; img_u_w: weakly augmented unlabeled data
            # img_u_s1, img_u_s2: strongly augmented unlabeled data; cutmix_box1, cutmix_box2: cutmix boxes
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            img_u_s1_copy = img_u_s1.clone()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            img_u_w_mixup, mixup_lambda = apply_mixup(img_u_w, img_u_w_mix)
            model.train()
            pred_x = model(img_x)  # img_x:[16, 1, 256, 256]
            pred_u_w = model(img_u_w).detach()
            pred_u_w_mixup = model(img_u_w_mixup)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)  # Results of two strong augmentations
            pred_u_w_softmax = pred_u_w.softmax(dim=1)
            # Dynamic threshold calculation
            uncertainty_entropy_map = compute_entropy_uncertainty_torch(pred_u_w_softmax)
            pred_u_w_dynamic_rate = UDAT(uncertainty_entropy_map, k=0.2)
            # Calculate mask region - directly use PyTorch tensor
            pred_u_w_ignore_mask = (uncertainty_entropy_map > pred_u_w_dynamic_rate).float()
            mask_unl = pred_u_w_ignore_mask.int()
            mask_unl = mask_unl.unsqueeze(1)
            # Entropy calculation for labeled data
            pred_x_softmax = pred_x.detach().softmax(dim=1)
            # Calculate entropy - use PyTorch version
            uncertainty_x_entropy_map = compute_entropy_uncertainty_torch(pred_x_softmax)
            pred_x_dynamic_rate = UDAT(uncertainty_x_entropy_map, k=0.2)
            # Calculate mask region - directly use PyTorch tensor
            pred_x_ignore_mask = (uncertainty_x_entropy_map > pred_x_dynamic_rate).int()
            mask_l = pred_x_ignore_mask.unsqueeze(1)
            mask_u_w = pred_u_w.argmax(dim=1)
            conf_u_w = torch.max(pred_u_w_softmax, dim=1)[0]
            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = apply_mask_cutmix(
                mask_u_w_mix, conf_u_w_mix, mask_u_w, conf_u_w, cutmix_box1)
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = apply_mask_cutmix(
                mask_u_w_mix, conf_u_w_mix, mask_u_w, conf_u_w, cutmix_box2)
            mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
            mask_u_w_mixup = apply_mask_mixup(mask_u_w, mask_u_w_mix, mixup_lambda, args)
            pred_pos = pred_u_w_mixup[:, 1, :, :]  # [B, H, W]
            # Loss for labeled data
            loss_sup_ori = (criterion_ce(pred_x, mask_x) + criterion_dice(pred_x.softmax(dim=1),
                                                                    mask_x.unsqueeze(1).float())) / 2.0
            # 20250421
            # Before calling exchange_patches/BSCE, ensure dimensions are correct
            # Check dimensions of mask_x and mask_u_w, add channel dimension only when needed
            if mask_x.dim() == 3:  # If it's [B, H, W]
                mask_x = mask_x.unsqueeze(1)  # Change to [B, 1, H, W]
            if mask_u_w.dim() == 3:  # If it's [B, H, W]
                mask_u_w = mask_u_w.unsqueeze(1)  # Change to [B, 1, H, W]
            # Call exchange_patches, all input dimensions are now [B, 1, H, W]
            new_image_l, new_gt, new_image_un_w, new_image_un_s, pse_label_temp \
                = exchange_patches(image_l=img_x, gt=mask_x, image_un_w=img_u_w, image_un_s=img_u_s1_copy,
                                   mask_l=mask_l, mask_unl=mask_unl, pse_label=mask_u_w,
                                   patch_size=16, exchange_ratio=0.5, random_seed=42)
            # Predict on exchanged data
            new_pred_x = model(new_image_l)  # Prediction on new labeled data
            new_pred_u_w = model(new_image_un_w).detach()  # Prediction on new weakly augmented unlabeled data
            new_pred_u_s = model(new_image_un_s)  # Prediction on new strongly augmented unlabeled data
            pred_img_u_s1_copy = model(img_u_s1_copy).detach()
            # Get pseudo-label and add channel dimension
            new_pse_label = new_pred_u_w.argmax(dim=1).unsqueeze(1)  # [B, 1, H, W]
            # Replace non -1 values in pse_label_temp into new_pse_label
            valid_mask = (pse_label_temp > -0.5)  # Use > -0.5 instead of != -1 to handle potential floating point errors
            new_pse_label[valid_mask] = pse_label_temp[valid_mask]
            # Calculate new losses
            loss_sup_new = (criterion_ce(new_pred_x, new_gt.squeeze(1)) +
                          criterion_dice(new_pred_x.softmax(dim=1), new_gt.float())) / 2.0
            loss_pse = nn.BCEWithLogitsLoss()(pred_pos, mask_u_w_mixup) \
                       + criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                        ignore=pred_u_w_ignore_mask) \
                       + criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                        ignore=pred_u_w_ignore_mask)
            loss_pse_new = criterion_dice(new_pred_u_s.softmax(dim=1), new_pse_label.float())
            loss_sup = loss_sup_ori + loss_sup_new
            loss_unsup = loss_pse + loss_pse_new
            loss = (loss_sup + loss_unsup) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.update(loss.item())
            # Calculate mask_ratio
            mask_ratio = (pred_u_w_ignore_mask == 0).sum() / pred_u_w_ignore_mask.numel()
            total_mask_ratio.update(mask_ratio.item())
            iters = epoch * len(trainloader_u) + i
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            writer.add_scalar('train/loss_all', loss.item(), iters)
            if (i % (len(trainloader_u) // 8) == 0):
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        model.eval()
        # Add evaluation metrics
        dice_medpy_class = [0] * (args.nclass - 1)
        hausdorff_class = [0] * (args.nclass - 1)
        hd95_class = [0] * (args.nclass - 1)
        iou_class = [0] * (args.nclass - 1)
        precision_class = [0] * (args.nclass - 1)
        recall_class = [0] * (args.nclass - 1)
        specificity_class = [0] * (args.nclass - 1)
        asd_class = [0] * (args.nclass - 1)
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()
                h, w = img.shape[-2:]
                img = F.interpolate(img, (args.crop_size, args.crop_size), mode='bilinear', align_corners=False)
                # img.shape=[1,10,256,256], HW size changed
                if args.data_mode == 'L':
                    img = img.permute(1, 0, 2, 3)
                # 3D image, treat the third dimension of ACDC as batch dimension, and the batch dimension of dataloader as channel dimension.
                # RGB image, no need for such operation
                pred = model(img)  # [10, 4, 256, 256](batch_size, nclass, h, w)
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)  # Restore HW size
                pred = pred.argmax(dim=1).unsqueeze(0)  # Remove channel dimension
                pred_i_numpy = pred.cpu().numpy()
                mask_i_numpy = mask.cpu().numpy()
                # Call function to calculate metrics: calculate_batch_metrics, calculate_metrics and update_metrics_lists should be used together
                calculate_batch_metrics(
                    pred_i_numpy, mask_i_numpy, args,
                    dice_medpy_class, hausdorff_class, hd95_class,
                    iou_class, precision_class, recall_class,
                    specificity_class, asd_class)
        dice_medpy_class = [dice_medpy * 100.0 / len(valloader) for dice_medpy in dice_medpy_class]
        mean_medpy_dice, mean_hd, mean_hd95, mean_iou, mean_precision, mean_recall, mean_specificity, mean_asd = calculate_metrics(
            None, None, args, valloader,
            dice_medpy_class, hausdorff_class, hd95_class,
            iou_class, precision_class, recall_class,
            specificity_class, asd_class)
        # Add results to the list
        update_metrics_lists(
            mean_medpy_dice, mean_hd, mean_hd95, mean_iou,
            mean_precision, mean_recall, mean_specificity, mean_asd,
            medpy_dice_list, hd_list, hd95_list, iou_list,
            precision_list, recall_list, specificity_list, asd_list)
        for (cls_idx, dice) in enumerate(dice_medpy_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                        '{:.2f}'.format(cls_idx, CLASSES[args.dataset][cls_idx], dice))
        logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_medpy_dice))
        is_best = mean_medpy_dice > previous_best
        previous_best = max(mean_medpy_dice, previous_best)
        checkpoint = {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'epoch': epoch, 'previous_best': previous_best, }
        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(snapshot_path, 'best.pth'))
            # Save only the model file, for testing
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'bestmodel_only.pth'))
    # # Add code: Save to excel
    estimation_list = np.vstack((medpy_dice_list, hd_list, hd95_list, iou_list, precision_list, recall_list,
                                 specificity_list, asd_list))
    estimation_list = estimation_list.T
    for i in range(len(estimation_list)):
        temp = estimation_list[i].tolist()
        ws1.append(temp)
    writer.close()
    # Save recorded results to the specified path's excel
    record_path = "experiment/{}/{}/record.xlsx".format(
        current_experiment_time, args.save_path)
    wb.save(record_path)
if __name__ == '__main__':
    main()
