import numpy as np
from medpy import metric
from model.unet import FEUNet


def get_model(args):
    """
    Get the model instance.
    Args:
        args: Arguments.
    Returns:
        model: The main model.
    """
    # Create the model, setting the input channel count based on the data mode
    in_chns = 3 if args.data_mode == 'RGB' else 1

    # Create the model
    model = FEUNet(in_chns=in_chns, class_num=args.nclass)
    return model


def apply_mask_cutmix(mask_source, conf_source, mask_target, conf_target, cutmix_box):
    """
    Apply the cutmix operation to the masks.

    Args:
        mask_source: Source mask.
        conf_source: Source confidence.
        mask_target: Target mask.
        conf_target: Target confidence.
        cutmix_box: A mask indicating the replacement region, where 1 means the region needs to be replaced.

    Returns:
        mask_mixed: Mixed mask.
        conf_mixed: Mixed confidence.
    """
    # Create copies of the target mask and confidence
    mask_mixed = mask_target.clone()
    conf_mixed = conf_target.clone()

    # Apply cutmix
    mask_mixed[cutmix_box == 1] = mask_source[cutmix_box == 1]
    conf_mixed[cutmix_box == 1] = conf_source[cutmix_box == 1]

    return mask_mixed, conf_mixed


def calculate_batch_metrics(pred_i_numpy, mask_i_numpy, args,
                            dice_medpy_class, hausdorff_class, hd95_class,
                            iou_class, precision_class, recall_class,
                            specificity_class, asd_class):
    """
    Calculate evaluation metrics for a single batch of medical images and accumulate them into corresponding lists.

    Args:
        pred_i_numpy: Numpy array of prediction results.
        mask_i_numpy: Numpy array of ground truth labels.
        args: Arguments object, containing configurations like nclass.
        dice_medpy_class: List to accumulate Dice coefficients.
        hausdorff_class: List to accumulate Hausdorff distances.
        hd95_class: List to accumulate 95th percentile Hausdorff distances.
        iou_class: List to accumulate IoU.
        precision_class: List to accumulate precision.
        recall_class: List to accumulate recall.
        specificity_class: List to accumulate specificity.
        asd_class: List to accumulate average surface distances.
    """
    # Iterate over each class (skipping the background class)
    for cls in range(1, args.nclass):
        # Calculate Dice coefficient
        dice_medpy = metric.binary.dc(pred_i_numpy == cls, mask_i_numpy == cls)
        dice_medpy_class[cls - 1] += dice_medpy

        # Calculate Hausdorff distance
        hd = metric.binary.hd(pred_i_numpy == cls, mask_i_numpy == cls)
        hausdorff_class[cls - 1] += hd

        # Calculate 95th percentile Hausdorff distance
        hd95 = metric.binary.hd95(pred_i_numpy == cls, mask_i_numpy == cls)
        hd95_class[cls - 1] += hd95

        # Calculate IoU (Jaccard coefficient)
        iou = metric.binary.jc(pred_i_numpy == cls, mask_i_numpy == cls)
        iou_class[cls - 1] += iou

        # Calculate precision
        precision = metric.binary.precision(pred_i_numpy == cls, mask_i_numpy == cls)
        precision_class[cls - 1] += precision

        # Calculate recall
        recall = metric.binary.recall(pred_i_numpy == cls, mask_i_numpy == cls)
        recall_class[cls - 1] += recall

        # Calculate specificity
        specificity = metric.binary.specificity(pred_i_numpy == cls, mask_i_numpy == cls)
        specificity_class[cls - 1] += specificity

        # Calculate average surface distance
        asd = metric.binary.asd(pred_i_numpy == cls, mask_i_numpy == cls)
        asd_class[cls - 1] += asd


def update_metrics_lists(mean_medpy_dice, mean_hd, mean_hd95, mean_iou,
                         mean_precision, mean_recall, mean_specificity, mean_asd,
                         medpy_dice_list, hd_list, hd95_list, iou_list,
                         precision_list, recall_list, specificity_list, asd_list):
    """
    Append the calculated evaluation metrics to their respective lists.

    Args:
        mean_medpy_dice: Mean Dice coefficient.
        mean_hd: Mean Hausdorff distance.
        mean_hd95: Mean 95th percentile Hausdorff distance.
        mean_iou: Mean IoU.
        mean_precision: Mean precision.
        mean_recall: Mean recall.
        mean_specificity: Mean specificity.
        mean_asd: Mean average surface distance.
        medpy_dice_list: List of Dice coefficients.
        hd_list: List of Hausdorff distances.
        hd95_list: List of 95th percentile Hausdorff distances.
        iou_list: List of IoU.
        precision_list: List of precision values.
        recall_list: List of recall values.
        specificity_list: List of specificity values.
        asd_list: List of average surface distances.
    """
    # Append the calculated evaluation metrics to their respective lists
    medpy_dice_list.append(mean_medpy_dice)
    hd_list.append(mean_hd)
    hd95_list.append(mean_hd95)
    iou_list.append(mean_iou)
    precision_list.append(mean_precision)
    recall_list.append(mean_recall)
    specificity_list.append(mean_specificity)
    asd_list.append(mean_asd)


def calculate_metrics(pred_i_numpy, mask_i_numpy, args, valloader,
                      dice_medpy_class, hausdorff_class, hd95_class,
                      iou_class, precision_class, recall_class,
                      specificity_class, asd_class):
    """
    Calculate the average values of various medical image segmentation evaluation metrics.

    Args:
        pred_i_numpy: Numpy array of prediction results (not used when calculating averages).
        mask_i_numpy: Numpy array of ground truth labels (not used when calculating averages).
        args: Arguments.
        valloader: Validation data loader.
        dice_medpy_class: List of Dice coefficients.
        hausdorff_class: List of Hausdorff distances.
        hd95_class: List of 95th percentile Hausdorff distances.
        iou_class: List of IoU values.
        precision_class: List of precision values.
        recall_class: List of recall values.
        specificity_class: List of specificity values.
        asd_class: List of average surface distances.

    Returns:
        mean_medpy_dice: Mean Dice coefficient.
        mean_hd: Mean Hausdorff distance.
        mean_hd95: Mean 95th percentile Hausdorff distance.
        mean_iou: Mean IoU.
        mean_precision: Mean precision.
        mean_recall: Mean recall.
        mean_specificity: Mean specificity.
        mean_asd: Mean average surface distance.
    """
    # Calculate averages
    # Note: dice_medpy_class has already been multiplied by 100.0 / len(valloader) before calling
    mean_medpy_dice = sum(dice_medpy_class) / len(dice_medpy_class)

    hausdorff_class = [hd / len(valloader) for hd in hausdorff_class]
    mean_hd = sum(hausdorff_class) / len(hausdorff_class)

    hd95_class = [hd95 / len(valloader) for hd95 in hd95_class]
    mean_hd95 = sum(hd95_class) / len(hd95_class)

    iou_class = [iou * 100.0 / len(valloader) for iou in iou_class]
    mean_iou = sum(iou_class) / len(iou_class)

    precision_class = [precision * 100.0 / len(valloader) for precision in precision_class]
    mean_precision = sum(precision_class) / len(precision_class)

    recall_class = [recall * 100.0 / len(valloader) for recall in recall_class]
    mean_recall = sum(recall_class) / len(recall_class)

    specificity_class = [specificity * 100.0 / len(valloader) for specificity in specificity_class]
    mean_specificity = sum(specificity_class) / len(specificity_class)

    asd_class = [asd / len(valloader) for asd in asd_class]
    mean_asd = sum(asd_class) / len(asd_class)

    return (mean_medpy_dice, mean_hd, mean_hd95, mean_iou,
            mean_precision, mean_recall, mean_specificity, mean_asd)


def apply_mixup(img_a, img_b, alpha=0.2):
    """
    Apply the mixup operation, blending two images proportionally.

    Args:
        img_a: The first image.
        img_b: The second image.
        alpha: Parameter for the Beta distribution, controlling the mixing ratio.

    Returns:
        img_mixed: The mixed image.
        lam: The mixing ratio.
    """
    # Sample the mixing ratio from the Beta distribution
    lam = np.random.beta(alpha, alpha)
    # Ensure lam is within the range [0, 1]
    lam = max(0, min(1, lam))

    # Mix the images
    img_mixed = lam * img_a + (1 - lam) * img_b

    return img_mixed, lam


def apply_mask_mixup(mask_a, mask_b, lam, args):
    """
    Apply the mixup operation to masks to generate mixed masks.
    Args:
        mask_a: First mask [B, H, W].
        mask_b: Second mask [B, H, W].
        lam: Mixing ratio.
        args: Arguments object, containing configurations like nclass.

    Returns:
        mixed_mask: The mixed mask [B, H, W].
    """

    mixed_mask = lam * mask_a.float() + (1 - lam) * mask_b.float()
    return mixed_mask
