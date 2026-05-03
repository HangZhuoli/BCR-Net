from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import torch

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, '..'))

from lib.opts import opts
from lib.detectors.ctdet_detector import CtdetDetector as Detector
from lib.datasets.dataset_factory import dataset_factory
from lib.logger import Logger

def compute_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def filter_results_by_iou(results, iou_thresh=0.3):
    filtered_results = {}

    for cls_id in results:
        dets = results[cls_id]
        if len(dets) == 0:
            continue

        # 按 score 从高到低排序
        dets = sorted(dets, key=lambda x: x[4], reverse=True)
        keep = []

        for det in dets:
            keep_flag = True
            for kept in keep:
                if compute_iou_xyxy(det[:4], kept[:4]) > iou_thresh:
                    keep_flag = False
                    break
            if keep_flag:
                keep.append(det)

        filtered_results[cls_id] = keep

    return filtered_results

# ---------------------------
# 绘制框
# ---------------------------
def draw_boxes(image, boxes, box_color=(0,255,0), thickness=2):
    vis_img = image.copy()
    for bbox in boxes:
        x, y, w, h = bbox
        cv2.rectangle(vis_img, (int(x), int(y)), (int(x+w), int(y+h)), box_color, thickness)
    return vis_img

def draw_pred_boxes(image, pred_results, score_thresh=0.5, box_color=(0,0,255), thickness=2):
    vis_img = image.copy()
    count = 0
    for cls_id in pred_results:
        for det in pred_results[cls_id]:
            x1, y1, x2, y2, score = det[:5]
            if score < score_thresh:
                continue
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, thickness)
            count = count+1
    print(count)
    return vis_img

# ---------------------------
# 保存图片
# ---------------------------
def save_image(opt, img, suffix, img_name):
    save_root = os.path.join('./vis_single_results', opt.exp_id)
    os.makedirs(save_root, exist_ok=True)
    out_path = os.path.join(save_root, f"{img_name}_{suffix}.jpg")
    cv2.imwrite(out_path, img)
    print(f"✔ 图片已保存: {out_path}")

# ---------------------------
# 单张图片检测
# ---------------------------
def test_single_separate(opt):
    # CPU/GPU
    if opt.gpus[0] >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpus[0])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 初始化 Dataset + heads
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Logger(opt)

    # load model
    detector = Detector(opt)

    img_name = os.path.splitext(os.path.basename(opt.input_img))[0]

    # ------------------ 读取 sharp 和 blur ------------------
    sharp_img_path = os.path.join(opt.sharp_data_dir, os.path.relpath(opt.input_img, opt.blur_data_dir))
    assert os.path.exists(sharp_img_path), f"Sharp 图像不存在: {sharp_img_path}"
    sharp_img = cv2.imread(sharp_img_path)
    blur_img = cv2.imread(opt.input_img)

    # ------------------ 获取 GT 框 ------------------
# ------------------ 获取 GT 框 ------------------
    DatasetObj = Dataset(opt, 'val')
    img_basename = os.path.basename(opt.input_img)  # 只取文件名
    img_id = None
    for iid in DatasetObj.images:
        info = DatasetObj.coco.loadImgs([iid])[0]
        if os.path.basename(info['file_name']) == img_basename:
            img_id = iid
            break
    assert img_id is not None, f"图像 ID 未找到: {opt.input_img}"
    
    ann_ids = DatasetObj.coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    anns = DatasetObj.coco.loadAnns(ann_ids)
    gt_boxes = [ann['bbox'] for ann in anns]  # [x, y, w, h]

    # ------------------ 绘制 GT 框 ------------------
    sharp_vis = draw_boxes(sharp_img, gt_boxes, box_color=(0,255,0))
    blur_vis  = draw_boxes(blur_img, gt_boxes, box_color=(0,255,0))

    save_image(opt, sharp_vis, 'sharp', img_name)
    save_image(opt, blur_vis, 'blur', img_name)

    # ------------------ 模型预测 ------------------

    ret = detector.run(blur_img)
    iou_thresh = 0.6     # IoU 限制
    score_thresh = 0.36     # 置信度限制
    # print(ret['results'])
    vis_results = filter_results_by_iou(ret['results'], iou_thresh)
    pred_vis = draw_pred_boxes(blur_img,vis_results,score_thresh=score_thresh,box_color=(0,0,255))
    save_image(opt, pred_vis, 'pred', img_name)
    
    # ret = detector.run(blur_img)
    # pred_vis = draw_pred_boxes(blur_img, ret['results'], box_color=(0,0,255))
    # save_image(opt, pred_vis, 'pred', img_name)

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    opt = opts().parse()

    if not hasattr(opt, 'input_img') or opt.input_img == '':
        print("❌ 必须指定 --input_img <path_to_image>")
        exit()
    if not hasattr(opt, 'sharp_data_dir') or not hasattr(opt, 'blur_data_dir'):
        print("❌ 必须指定 --sharp_data_dir 和 --blur_data_dir")
        exit()

    test_single_separate(opt)
