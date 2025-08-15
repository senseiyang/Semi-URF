import argparse
import os
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from openpyxl import Workbook
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
np.set_printoptions(linewidth=300)
from model.unet import FEUNet
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=5,
                    help='labeled data')
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    hd = metric.binary.hd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    IOU = metric.binary.jc(pred, gt)
    Precision = metric.binary.precision(pred, gt)
    Recall = metric.binary.recall(pred, gt)
    Specificity = metric.binary.specificity(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd, hd95, IOU, Precision, Recall, Specificity, asd

def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "experiment/test"
    test_save_path = "experiment/test/predictions/"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = FEUNet(in_chns=1, class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, 'bestmodel_only.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    net = net.cuda()
    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    metric_save_xlsx = "experiment/test/prediction.xlsx"
    wb = Workbook()
    ws1 = wb.create_sheet('test_metric', 0)
    ws1.append(['dice', 'hd', 'hd95', 'IOU', 'Precision', 'Recall', 'Specificity', 'asd'])
    ws1.append(metric[0].tolist())
    ws1.append(metric[1].tolist())
    ws1.append(metric[2].tolist())
    ws1.append(((metric[0]+metric[1]+metric[2])/3).tolist())
    wb.save(metric_save_xlsx)
