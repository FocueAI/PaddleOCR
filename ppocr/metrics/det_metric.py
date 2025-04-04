# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["DetMetric", "DetFCEMetric"]

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]  # .shape=[1,7,4,2]
        ignore_tags_batch = batch[3]  # .shape=[1,7]
        gt_classes_batch = batch[4]  # add 2025-3-31  .shape=[1,7] 
        for pred, gt_polyons, ignore_tags, gt_classes in zip(  # 将 batch 遍历出 一张张 图像
            preds, gt_polyons_batch, ignore_tags_batch, gt_classes_batch
        ):
            # ---------------------------------- prepare gt
            # gt_info_list = [
            #     {"points": gt_polyon, "text": "", "ignore": ignore_tag, "cls": gt_class}
            #     for gt_polyon, ignore_tag, gt_class in zip(gt_polyons, ignore_tags, gt_cls)
            # ]                             #   .shape=[7,4,2], .shape=[7,]
            gt_info_list = []   #   .shape =             [7,4,2],   [7,]         [7,]
            for gt_polyon, ignore_tag, gt_class in zip(gt_polyons, ignore_tags, gt_classes):
                 gt_info_list.append({"points": gt_polyon, "text": "", "ignore": ignore_tag, "cls": gt_class})
            
            
            
            
            # ----------------------------------- prepare det
            # det_info_list = [
            #     {"points": det_polyon, "text": ""} for det_polyon in pred["points"]
            # ]
            # det_info_list = [
            #     {"points": det_polyon['points'], "text": "", "cls": det_polyon["classes"]} for det_polyon in pred
            # ]
            det_info_list = []
            for index, det_poly in enumerate(pred["points"]):
                det_info_list.append({"points": det_poly, "text": "", "score": pred["scores"][index], "cls": pred["classes"][index]})
            
            
            
            
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)  #TODO: 明天继续改造该函数!!!!!
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results


class DetFCEMetric(object):
    def __init__(self, main_indicator="hmean", **kwargs):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        """
        batch: a list produced by dataloaders.
            image: np.ndarray  of shape (N, C, H, W).
            ratio_list: np.ndarray  of shape(N,2)
            polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
        preds: a list of dict produced by post process
             points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
        """
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polyons, ignore_tags in zip(
            preds, gt_polyons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polyon, "text": "", "ignore": ignore_tag}
                for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polyon, "text": "", "score": score}
                for det_polyon, score in zip(pred["points"], pred["scores"])
            ]

            for score_thr in self.results.keys():
                det_info_list_thr = [
                    det_info
                    for det_info in det_info_list
                    if det_info["score"] >= score_thr
                ]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_thr].append(result)

    def get_metric(self):
        """
        return metrics {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        metrics = {}
        hmean = 0
        for score_thr in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_thr])
            # for key, value in metric.items():
            #     metrics['{}_{}'.format(key, score_thr)] = value
            metric_str = "precision:{:.5f} recall:{:.5f} hmean:{:.5f}".format(
                metric["precision"], metric["recall"], metric["hmean"]
            )
            metrics["thr {}".format(score_thr)] = metric_str
            hmean = max(hmean, metric["hmean"])
        metrics["hmean"] = hmean

        self.reset()
        return metrics

    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: [],
        }  # clear results
