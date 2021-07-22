import json
import os

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.evaluation import Metric, GenericPluginMetric
from avalanche.training.plugins.evaluation import EvaluationPlugin

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from typing import Optional, List
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import datetime

import haitain_detection as haitain

categories = {
    1: "Pedestrain",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle"
}


def collate_fn(batch):
    return list(zip(*batch))


def empty(*args, **kwargs):
    return torch.tensor(0.0)


def detection_metrics(gt_path, stream=False, experience=False, store=False):
    metrics = []
    if stream:
        metrics.append(DetectionEvaluationPlugin(gt_path=gt_path, reset_at='stream', emit_at='stream',
                                                 mode='eval', name='CocoEvalStream', store=store))
    if experience:
        metrics.append(DetectionEvaluationPlugin(gt_path=gt_path, reset_at='experience', emit_at='experience',
                                                 mode='eval', name='CocoEvalExp', store=store))
    return metrics


class DetectionEvaluationPlugin(GenericPluginMetric):

    def __init__(self, reset_at, emit_at, mode, gt_path, name=None, store=None):
        self._detection_results = DetectionMetric(gt_path, store=store)
        self.name = name
        super(DetectionEvaluationPlugin, self).__init__(
            self._detection_results, reset_at=reset_at, emit_at=emit_at,
            mode=mode)

    def update(self, strategy):
        img_ids = [target['image_id'] for target in strategy.mb_y]
        self._detection_results.update(strategy.mb_output, img_ids)

    def __str__(self):
        return "CocoEval" if self.name is None else self.name


class DetectionMetric(Metric):

    def __init__(self, gt_path, store=False):
        self.gt_path = gt_path
        self.store = store
        self._gt = self.load_gt()
        self._dt = []

    @torch.no_grad()
    def update(self, prediction, img_ids):
        for img_id, img_pred in zip(img_ids, prediction):
            for box, label, score in zip(img_pred['boxes'], img_pred['labels'], img_pred['scores']):
                box = box.tolist()
                box_hw = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                self._dt.append({
                    'image_id': int(img_id),
                    'category_id': int(label),
                    'bbox': box_hw,
                    'score': float(score)
                })

    def result(self) -> Optional:
        with open("./tmp_results.json", "w") as f:
            json.dump(self._dt, f)
        dt = self._gt.loadRes('./tmp_results.json')
        coco_eval = COCOeval(self._gt, dt, 'bbox')
        coco_eval.params.iouThrs = [0.5, 0.7]
        coco_eval.evaluate()
        coco_eval.accumulate()

        out = "\n"
        base_str = "Average Precision (AP) @[ Category={} \t| IoU={} | area=all | maxDets=100 ] = {:.3f}"
        ious = [0, 0, 1, 1, 1, 0]

        mean, count = 0, 0
        for i in range(1, 7):
            s = coco_eval.eval['precision'][ious[i - 1], :, i - 1, 0, -1]
            if len(s[s == -1] == 0):
                res = -1
            else:
                res = np.mean(s[s > -1])
                mean += res
                count += 1
            out += base_str.format(categories[i], coco_eval.params.iouThrs[ious[i - 1]], res)
            out += "\n"
        out += base_str.format("all", "0.5/0.7", mean / count)
        out += "\n"
        os.remove('./tmp_results.json')

        self.store_results()
        return out

    def store_results(self):
        if self.store is not None and len(self._dt) > 0:
            file_name = f"{self.store}.json"
            with open(f"./det_results/{file_name}", 'a') as f:
                json.dump(self._dt, f)

    def reset(self) -> None:
        self._dt = []

    def load_gt(self):
        return COCO(self.gt_path)


class DetectionBaseStrategy(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion, evaluator: EvaluationPlugin,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 eval_every=-1):

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """
        Necessary (?) because default collate_fn doesn't work with detection datasets.
        Maybe there's a better way of doing this.
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.train_mb_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    def make_eval_dataloader(self, num_workers=0, pin_memory=True,
                             **kwargs):
        """
        Necessary (?) because default collate_fn doesn't work with detection datasets.
        Maybe there's a better way of doing this.
        """
        self.dataloader = DataLoader(
            self.adapted_dataset,
            batch_size=self.eval_mb_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    def _unpack_minibatch(self):
        '''
        Necessary because in detection, x and y aren't
        tensors but lists (of tensors, and dicts)
        '''
        assert len(self.mbatch) >= 3
        self.mbatch[0] = list(sample.to(self.device) for sample in self.mbatch[0])
        self.mbatch[1] = [{k: v.to(self.device) for k, v in t.items()} for t in self.mbatch[1]]
        self.mbatch[-1] = torch.tensor(self.mbatch[-1]).to(self.device)
        # TODO: there could be more tensors in the mbatch

    def forward(self):
        """
        Override necessary because the torchvision models
        calculate losses inside the forward call of the model.
        """
        if self.is_training:
            loss_dict = self.model(self.mb_x, self.mb_y)
            self.loss += sum(loss_value for loss_value in loss_dict.values())
            return loss_dict
        elif self.is_eval:
            return self.model(self.mb_x)

def create_train_val_set(root, validation_proportion=0.1):
    splits = ['train', 'val', 'val', 'val']
    task_dicts = [{'city': 'Shanghai', 'location': 'Citystreet', 'period': 'Daytime', 'weather': 'Clear'},
                  {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
                  {'period': 'Night'},
                  {'period': 'Daytime', 'weather': 'Rainy'}]

    match_fns = [haitain.create_match_fn_from_dict(td) for td in task_dicts]
    train_sets = [haitain.get_matching_set(root, split, match_fn, haitain.get_transform(True)) for
                  match_fn, split in zip(match_fns, splits)]
    val_sets = []
    for ts in train_sets:
        cut_off = int((1.0 - validation_proportion) * len(ts))
        all_samples = ts.samples
        ts.samples = all_samples[:cut_off]
        val_samples = all_samples[cut_off:]
        val_sets.append(haitain.HaitainDetectionSet(ts.root, val_samples, ts.annotation_file, ts.transform,
                                                    ts.meta))

    return [AvalancheDataset(train_set) for train_set in train_sets], \
           [AvalancheDataset(val_set) for val_set in val_sets]


def create_test_set(root):
    task_dicts = [{'location': ['Citystreet', 'Countryroad'], 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
                  {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
                  {'period': 'Night'},
                  {'period': 'Daytime', 'weather': 'Rainy'}]

    match_fns = [haitain.create_match_fn_from_dict(td) for td in task_dicts]
    test_sets = [haitain.get_matching_set(root, 'test', match_fn, haitain.get_transform(False)) for
                 match_fn in match_fns]

    test_set_keys = ["Task 1", "Task 2", "Task 3", "Task 4"]
    test_sets = [AvalancheDataset(test_set) for test_set in test_sets if len(test_set) > 0]

    return test_sets, test_set_keys
