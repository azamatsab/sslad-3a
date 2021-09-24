from torchvision import transforms
import numpy as np
import torch
from catalyst.data.sampler import BalanceClassSampler, BatchBalanceClassSampler
import albumentations as A

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils import AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader, GroupBalancedDataLoader, ReplayDataLoader

from transform_wrapper import Atw

"""
A strategy pulgin can change the strategy parameters before and after 
every important fucntion. The structure is shown belown. For instance, self.before_train_forward() and
self.after_train_forward() will be called before and after the main forward loop of the strategy.

The training loop is organized as follows::
        **Training loop**
            train
                train_exp  # for each experience
                    adapt_train_dataset
                    train_dataset_adaptation
                    make_train_dataloader
                    train_epoch  # for each epoch
                        train_iteration # for each minibatch
                            forward
                            backward
                            model update

        **Evaluation loop**
        The evaluation loop is organized as follows::
            eval
                eval_exp  # for each experience
                    adapt_eval_dataset
                    eval_dataset_adaptation
                    make_eval_dataloader
                    eval_epoch  # for each epoch
                        eval_iteration # for each minibatch
                            forward
                            backward
                            model update
"""

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ClassStrategyPlugin(StrategyPlugin):

    def __init__(self):
        super(ClassStrategyPlugin).__init__()
        self.exp_num = 0
        self.ext_mem = None
        self.bad_prev = False

    def before_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_exp(self, strategy: 'BaseStrategy', num_workers=0, shuffle=False,
                              pin_memory=True, **kwargs):
        targets = np.array(strategy.adapted_dataset.targets)
        if self.exp_num > 0:
            # strategy.optimizer = torch.optim.SGD(strategy.model.parameters(), lr=0.001)
            strategy._criterion = torch.nn.CrossEntropyLoss()

        self.exp_num += 1

    def before_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                        **kwargs):
        # targets = np.array(strategy.experience.dataset.targets)
        # dataset = strategy.experience.dataset

        # indices = {}
        # existed_targets = list(set(strategy.experience.dataset.targets))
        # for index in range(len(existed_targets)):
        #     label = existed_targets[index]
        #     indices[index] = np.argwhere(targets == label).reshape(-1)
        # dataset.tasks_pattern_indices = indices
        pass

    def after_train_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        torchvision_transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomPerspective(),
                                transforms.RandomErasing(),
                                # Atw(A.HueSaturationValue(p=0.5)),
                                # Atw(A.RandomBrightness(p=0.5)),
                                # normalize,
                            ])
        strategy.adapted_dataset = strategy.adapted_dataset.add_transforms(torchvision_transform)
        # if self.exp_num == 0:
        #     datasets = []
        #     for _ in range(3):
        #         datasets.append(strategy.adapted_dataset._fork_dataset())
        #     strategy.adapted_dataset = AvalancheConcatDataset(datasets)

    def before_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_backward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_update(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_training(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_dataset_adaptation(self, strategy: 'BaseStrategy',
                                       **kwargs):
        pass

    def after_eval_dataset_adaptation(self, strategy: 'BaseStrategy', **kwargs):
        # torchvision_transform = transforms.Compose([
        #                         normalize,
        #                     ])
        # strategy.adapted_dataset = strategy.adapted_dataset.add_transforms(torchvision_transform)
        pass

    def before_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_exp(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def before_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_forward(self, strategy: 'BaseStrategy', **kwargs):
        pass

    def after_eval_iteration(self, strategy: 'BaseStrategy', **kwargs):
        pass
