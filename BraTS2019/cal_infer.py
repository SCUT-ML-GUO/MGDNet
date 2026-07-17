import os
import math
import time

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import nibabel as nib
from medpy import metric
from networks.MGDNet import *


def cal_dice(output, target):
    dice1 = metric.binary.dc(output == 3, target == 3)
    dice2 = metric.binary.dc((output == 1) | (output == 3), (target == 1) | (target == 3))
    dice3 = metric.binary.dc(output != 0, target != 0)
    return dice1, dice2, dice3

def calculate_metric_percase(pred, gt):
    # ET: label 3
    # TC: label 1 + 3
    # WT: label 1 + 2 + 3
    dice_ET, dice_TC, dice_WT = cal_dice(pred, gt)
    hd_ET, hd_TC, hd_WT = 0, 0, 0
    if np.any(pred == 3) and np.any(gt == 3):
        hd_ET = metric.binary.hd95(pred == 3, gt == 3)
    if np.any((pred == 1) | (pred == 3)) and np.any((gt == 1) | (gt == 3)):
        hd_TC = metric.binary.hd95((pred == 1) | (pred == 3), (gt == 1) | (gt == 3))
    if np.any(pred != 0) and np.any(gt != 0):
        hd_WT = metric.binary.hd95(pred != 0, gt != 0)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice_ET, dice_TC, dice_WT, jc, hd_ET, hd_TC, hd_WT, asd

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    start_time = time.perf_counter()
    print(image.shape)
    c, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:]).astype(np.float32)
    cnt = np.zeros(image.shape[1:]).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[:,xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(test_patch,axis=0).astype(np.float32)
                t1 = 2
                t1ce = 1
                t2 = 3
                flair = 0

                t1_t1ce = [t1, t1ce]
                t2_flair = [t2, flair]

                # t1t1ce + t2flair
                images1 = test_patch[:, t1_t1ce, :, :, :]
                images2 = test_patch[:, t2_flair, :, :, :]


                images1 = torch.from_numpy(images1).to(device)
                images2 = torch.from_numpy(images2).to(device)
                y1 = net(images1, images2)

                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]    
                
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    end_time = time.perf_counter()

    latency = (end_time - start_time) * 1000
    return label_map, score_map, latency

def test_all_case(device, net, image_list, num_classes=2, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None,file_name='Not Found File Name'):
    latencies = []
    total_metric = 0.0
    for ith,image_path in enumerate(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label == 4] = 3 
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map, latency = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        latencies.append(latency)
        print('ET in label: ', np.sum(label == 3))
        print('ET in prediction: ', np.sum(prediction == 3))
        print('prediction set: ', np.unique(prediction), ' label set: ', np.unique(label))

        print('pred shape: {}, label shape: {}', prediction.shape, label.shape) #############
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0,0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        print('ith: %d,\tdice_ET: %.5f,\tdice_TC: %.5f,\tdice_WT: %.5f,\tavg_dice: %.5f\tjaccard: %.5f,\thausdorff distance ET: %.5f,\thausdorff distance TC: %.5f,\thausdorff distance WT: %.5f,\tavg hd: %.5f\tasd: %.5f' % (ith+1, single_metric[0], single_metric[1], single_metric[2], (single_metric[0] + single_metric[1] + single_metric[2]) / 3, single_metric[3], single_metric[4], single_metric[5], single_metric[6], (single_metric[4] + single_metric[5] +single_metric[6]) / 3, single_metric[7]))
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%02d_pred.nii.gz"%(ith))
            nib.save(nib.Nifti1Image(image[0].astype(np.float32), np.eye(4)), test_save_path + "%02d_img.nii.gz"%(ith))
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz"%(ith))
    avg_metric = total_metric / len(image_list)
    print('metric: dice_ET\tdice_TC\tdice_WT\tjaccard\thausdorff distance\tasd\t')
    print('average metric is {}'.format(avg_metric))
    with open('test/'+file_name+'.txt', 'w') as f:   
        f.write('dice_ET: %.3f\tdice_TC: %.3f\tdice_WT: %.3f\tavg_dice: %.3f\n' % (avg_metric[0], avg_metric[1], avg_metric[2], (avg_metric[0] + avg_metric[1] + avg_metric[2]) / 3))
        f.write('hd_ET: %.3f\thd_TC: %.3f\thd_WT: %.3f\tavg_df: %.3f\n' % (avg_metric[4], avg_metric[5], avg_metric[6], (avg_metric[4] + avg_metric[5] + avg_metric[6]) / 3))
        f.write('jaccard: %.3f\tasd: %.3f\n' % (avg_metric[3], avg_metric[7]))

    latencies = np.array(latencies)

    print("\n" + "=" * 50)
    print("测量结果:")
    print("=" * 50)
    print(f"平均推理时间: {np.mean(latencies):.2f} ms")
    print(f"推理时间标准差: {np.std(latencies):.2f} ms")
    print(f"最小推理时间: {np.min(latencies):.2f} ms")
    print(f"最大推理时间: {np.max(latencies):.2f} ms")
    print(f"中位数推理时间: {np.median(latencies):.2f} ms")
    print(f"FPS (帧率): {1000 / np.mean(latencies):.2f}")

    peak_memory_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_memory_bytes / (1024 ** 3)  # 转为GB

    if peak_memory_gb is not None:
        print(f"峰值GPU内存占用: {peak_memory_bytes:,} bytes")
        print(f"峰值GPU内存占用: {peak_memory_gb:.3f} GB")
        
    return avg_metric



if __name__ == '__main__':
    file_name = "MGDNet"
    data_path = ''
    valid_txt = ''
    test_save_path = 'predictions/'+file_name+'/'  
    save_mode_path = 'results/'+file_name+'.pth'  
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = MGDNet(in_channels=2,num_classes=4).to(device) 

    net.load_state_dict(torch.load(save_mode_path)['model'])
    print("init weight from {}".format(save_mode_path))
    net.eval()
    with open(valid_txt, 'r') as f:
        image_list = [os.path.join(data_path, x.strip().split('/')[-1]) for x in f.readlines()]
    print('test images num: ',len(image_list))
    avg_metric = test_all_case(device, net, image_list, num_classes=4,
                                patch_size=(128,128,128), stride_xy=32, stride_z=16,
                                save_result=False,test_save_path=test_save_path,file_name=file_name)
