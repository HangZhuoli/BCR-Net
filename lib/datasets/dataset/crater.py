# -*- coding: utf-8 -*-
# 本文件用于处理 Crater（陨石坑）数据集（单类别版本）

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data


# ===========================
# Crater 数据集（单类别定义），同时定义对于数据集所采取的操作，类似于分批读取数据集的内部信息，实现网络的高效去模糊
# 实现对于数据集的预处理操作
# 用于对数据集进行数据预处理的操作，实现对于数据的进行加载迭代器等形式进行数据的读取
# ===========================
CRATER_num_classes = 1
CRATER_class_name = ['crater']
CRATER_valid_ids = [1]


class CraterDataset(data.Dataset):
    num_classes = CRATER_num_classes
    #将陨石坑图像进行二分插值放大，实现对于像素点补全来实现对于图像细节的补充
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        """
        参数：
            opt.sharp_data_dir : 清晰图像数据路径
            opt.blur_data_dir  : 模糊图像数据路径
            split              : 数据划分类型（train / val / test-dev） 为标签的路径
        """
        super(CraterDataset, self).__init__()
        self.sharp_data_dir = opt.sharp_data_dir
        self.blur_data_dir = opt.blur_data_dir

        # 图像路径
        self.sharp_img_dir = os.path.join(self.sharp_data_dir, f'Geo-{split}/images')
        self.blur_img_dir  = os.path.join(self.blur_data_dir,  f'Geo-{split}/images')

        # 标注路径，类别数量目前实际为1
        if split == 'test-dev':
            self.annot_path = os.path.join(
                '../dataset/Crater/Annotations/annotations' + str(CRATER_num_classes),
                'Geo-test-dev.json')
        else:
            #标注的命名方式，与此前保持一致，记住一定需要保持一致
            self.annot_path = os.path.join(
                '../dataset/Crater/Annotations/annotations' + str(CRATER_num_classes),
                f'Geo-{split}.json')

        print('annot_path:', self.annot_path)

        # 类别定义
        self.max_objs = 128
        self.class_name = CRATER_class_name
        self._valid_ids = CRATER_valid_ids
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        # PCA颜色扰动参数
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484,  0.41340352],
            [-0.5832747,   0.00994535, -0.81221408],
            [-0.56089297,  0.71832671,  0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt

        print(f'==> Initializing CraterDataset ({split}) ...')
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print(f'Loaded {self.num_samples} {split} samples')

    def _to_float(self, x):
        """浮点数格式化输出"""
        return float("{:.2f}".format(x))

    # ===========================
    # COCO 格式转换
    # ===========================
    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                # 由于只有一个类别，因此 cls_ind = 1 对应 crater
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    # ===========================
    # 保存检测结果
    # ===========================
    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open(f'{save_dir}/results.json', 'w'))

    # ===========================
    # 评估模块（基于 COCO API）
    # ===========================
    def run_eval(self, results, save_dir):
        # 添加识别的相关参数，因为在coco数据集合进行评估的时候，需要对于coco'数据集格式的完整性进行校验
        if 'info' not in self.coco.dataset:
            self.coco.dataset['info'] = {'description': 'Crater dataset'}
        if 'licenses' not in self.coco.dataset:
            self.coco.dataset['licenses'] = []
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes(f'{save_dir}/results.json')
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        #记录评估的结果数值，确定iou的数值,通过利用固定iou的数值来确定评估的结果
        coco_eval.params.iouThrs = [0.5]
        coco_eval.evaluate()
        coco_eval.accumulate()

        # 保存输出结果到文本
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = my_stdout = io.StringIO()
        coco_eval.summarize()
        sys.stdout = old_stdout
        results_text = my_stdout.getvalue()
        print(results_text)
        with open(os.path.join(save_dir, 'result.txt'), 'a') as f:
            f.write(results_text)
            f.write('\n')
