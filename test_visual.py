# 计算结果并可视化,这是1000张图片的可视化结果，对于验证数据集的可视化操作
"""
1、其中预测框为红色显示，而绿色代表陨石坑中的真实表示，对

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import torch
from progress.bar import Bar

import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, '..'))

from lib.opts import opts
from lib.logger import Logger
from lib.utils.utils import AverageMeter
from lib.datasets.dataset_factory import dataset_factory
from lib.detectors.ctdet_detector import CtdetDetector as Detector


# ---------------------------
# Dataset Prefetch
# ---------------------------
class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.coco = dataset.coco
        self.sharp_img_dir = dataset.sharp_img_dir
        self.blur_img_dir = dataset.blur_img_dir

        if opt.inp_sharp_or_blur == 'sharp':
            self.img_dir = self.sharp_img_dir
        else:
            self.img_dir = self.blur_img_dir

        self.pre_process_func = pre_process_func
        self.opt = opt
    
    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        images, meta = {}, {}
        for scale in self.opt.test_scales:
            images[scale], meta[scale] = self.pre_process_func(image, scale)

        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)



# ---------------------------
# Visualization Function
# ---------------------------
def save_visualization(opt, dataset, img_id, image, ret):
    results = ret['results']

    img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    img_name = os.path.splitext(img_info['file_name'])[0]

    save_root = os.path.join('./dataset/visual', opt.exp_id, img_name)
    os.makedirs(save_root, exist_ok=True)

    vis_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # 1. 绘制预测框（红色）
    score_thresh = 0.5  # 可调
    for cls_id in results:
        for det in results[cls_id]:
            x1, y1, x2, y2, score = det[:5]
            if score < score_thresh:
                continue
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
            cv2.putText(vis_img, f"{score:.2f}", (int(x1), int(y1)-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # 2. 绘制 GT（绿色）
    ann_ids = dataset.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    anns = dataset.coco.loadAnns(ann_ids)
    # anns = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=[img_id]))
    for ann in anns:
        x, y, w, h = ann['bbox']
        cv2.rectangle(vis_img, 
                      (int(x), int(y)),
                      (int(x + w), int(y + h)),
                      (0, 255, 0), 2)

    # 保存图像
    out_path = os.path.join(save_root, f"{img_name}_det.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


# ---------------------------
# Prefetch Testing with Visualization
# ---------------------------
def prefetch_test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    dataset = Dataset(opt, 'val' if not opt.trainval else 'test')
    detector = Detector(opt)
    
    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process), 
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    bar = Bar(opt.exp_id, max=len(dataset))

    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        ret = detector.run(pre_processed_images)

        raw_img = pre_processed_images['image'][0].numpy()
        save_visualization(opt, dataset, int(img_id.numpy()), raw_img, ret)

        results[int(img_id.numpy())] = ret['results']

        # progress bar
        Bar.suffix = f'[{ind}/{len(dataset)}]|Tot: {bar.elapsed_td} |ETA: {bar.eta_td}'
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix += f'|{t} {avg_time_stats[t].avg:.3f} '
        bar.next()

    bar.finish()
    dataset.run_eval(results, opt.save_dir)


# ---------------------------
# Normal Testing (no vis)
# ---------------------------
def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    Logger(opt)
    
    dataset = Dataset(opt, 'val' if not opt.trainval else 'test')
    detector = Detector(opt)

    results = {}
    bar = Bar(opt.exp_id, max=len(dataset))
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}

    for ind in range(len(dataset)):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]

        if opt.inp_sharp_or_blur == 'sharp':
            img_path = os.path.join(dataset.sharp_img_dir, img_info['file_name'])
        else:
            img_path = os.path.join(dataset.blur_img_dir, img_info['file_name'])

        ret = detector.run(img_path)
        results[img_id] = ret['results']

        Bar.suffix = f'[{ind}/{len(dataset)}]|Tot: {bar.elapsed_td} |ETA: {bar.eta_td}'
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix += f'|{t} {avg_time_stats[t].avg:.3f} '

        bar.next()

    bar.finish()
    dataset.run_eval(results, opt.save_dir)



# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    opt = opts().parse()
    if opt.not_prefetch_test:
        test(opt)
    else:
        prefetch_test(opt)
