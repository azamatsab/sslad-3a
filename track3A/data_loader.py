################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Avalanche supports data loading using pytorch's dataloaders.
    This module provides custom dataloaders for continual learning such as
    support for balanced dataloading between different tasks or balancing
    between the current data and the replay memory.
"""
from itertools import chain
from typing import Dict, Sequence

import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from catalyst.data.sampler import BalanceClassSampler, BatchBalanceClassSampler

from avalanche.benchmarks.utils import AvalancheDataset


def _default_collate_mbatches_fn(mbatches):
    """ Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually this
    is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = torch.cat([el[i] for el in mbatches], dim=0)
        batch.append(t)
    return batch


class TaskBalancedDataLoader:
    def __init__(self, data: AvalancheDataset,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 **kwargs):
        """ Task-balanced data loader for Avalanche's datasets.

        The iterator returns a mini-batch balanced across each task, which
        makes it useful when training in multi-task scenarios whenever data is
        highly unbalanced.

        If `oversample_small_tasks == True` smaller tasks are
        oversampled to match the largest task. Otherwise, once the data for a
        specific task is terminated, that task will not be present in the
        subsequent mini-batches.

        :param data: an instance of `AvalancheDataset`.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """
        self.data = data
        self.dataloaders: Dict[int, DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        # split data by task.
        task_datasets = []
        # print('Task Dataset', self.data.task_set)
        for task_label in self.data.task_set:
            tdata = self.data.task_set[task_label]
            task_datasets.append(tdata)

        # the iteration logic is implemented by GroupBalancedDataLoader.
        # we use kwargs to pass the arguments to avoid passing the same
        # arguments multiple times.
        if 'data' in kwargs:
            del kwargs['data']
        # needed if they are passed as positional arguments
        kwargs['oversample_small_groups'] = oversample_small_tasks
        kwargs['collate_mbatches'] = collate_mbatches
        self._dl = GroupBalancedDataLoader(datasets=task_datasets, **kwargs)

    def __iter__(self):
        for el in self._dl.__iter__():
            yield el

    def __len__(self):
        return self._dl.__len__()


class GroupBalancedDataLoader:
    def __init__(self, datasets: Sequence[AvalancheDataset],
                 oversample_small_groups: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 **kwargs):
        """ Data loader that balances data from multiple datasets.

        Mini-batches emitted by this dataloader are created by collating
        together mini-batches from each group. It may be used to balance data
        among classes, experiences, tasks, and so on.

        If `oversample_small_groups == True` smaller groups are oversampled to
        match the largest group. Otherwise, once data from a group is
        completely iterated, the group will be skipped.

        :param datasets: an instance of `AvalancheDataset`.
        :param oversample_small_groups: whether smaller groups should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param kwargs: data loader arguments used to instantiate the loader for
            each group separately. See pytorch :class:`DataLoader`.
        """
        self.datasets = datasets
        self.dataloaders = []
        self.oversample_small_groups = oversample_small_groups
        self.collate_mbatches = collate_mbatches

        for data in self.datasets:
            self.dataloaders.append(DataLoader(data, **kwargs))
        self.max_len = max([len(d) for d in self.dataloaders])

    def __iter__(self):
        iter_dataloaders = []
        for dl in self.dataloaders:
            iter_dataloaders.append(iter(dl))

        max_num_mbatches = max([len(d) for d in iter_dataloaders])
        for it in range(max_num_mbatches):
            mb_curr = []
            is_removed_dataloader = False
            # copy() is necessary because we may remove keys from the
            # dictionary. This would break the generator.
            for tid, t_loader in enumerate(iter_dataloaders):
                try:
                    batch = next(t_loader)
                except StopIteration:
                    # StopIteration is thrown if dataset ends.
                    if self.oversample_small_groups:
                        # reinitialize data loader
                        iter_dataloaders[tid] = iter(self.dataloaders[tid])
                        batch = next(iter_dataloaders[tid])
                    else:
                        # We iteratated over all the data from this group
                        # and we don't need the iterator anymore.
                        iter_dataloaders[tid] = None
                        is_removed_dataloader = True
                        continue
                mb_curr.append(batch)
            yield self.collate_mbatches(mb_curr)

            # clear empty data-loaders
            if is_removed_dataloader:
                while None in iter_dataloaders:
                    iter_dataloaders.remove(None)

    def __len__(self):
        return self.max_len


class ReplayDataLoader:
    def __init__(self, data: AvalancheDataset, memory: AvalancheDataset = None,
                 oversample_small_tasks: bool = False,
                 collate_mbatches=_default_collate_mbatches_fn,
                 batch_size: int = 32,
                 **kwargs):
        """ Custom data loader for rehearsal strategies.

        The iterates in parallel two datasets, the current `data` and the
        rehearsal `memory`, which are used to create mini-batches by
        concatenating their data together. Mini-batches from both of them are
        balanced using the task label (i.e. each mini-batch contains a balanced
        number of examples from all the tasks in the `data` and `memory`).
        
        If `oversample_small_tasks == True` smaller tasks are oversampled to
        match the largest task.

        :param data: AvalancheDataset.
        :param memory: AvalancheDataset.
        :param oversample_small_tasks: whether smaller tasks should be
            oversampled to match the largest one.
        :param collate_mbatches: function that given a sequence of mini-batches
            (one for each task) combines them into a single mini-batch. Used to
            combine the mini-batches obtained separately from each task.
        :param batch_size: the size of the batch. It must be greater than or
            equal to the number of tasks.
        :param kwargs: data loader arguments used to instantiate the loader for
            each task separately. See pytorch :class:`DataLoader`.
        """

        self.data = data
        self.memory = memory
        self.loader_data: Sequence[DataLoader] = {}
        self.loader_memory: Sequence[DataLoader] = {}
        self.oversample_small_tasks = oversample_small_tasks
        self.collate_mbatches = collate_mbatches

        import numpy as np
        targets = np.array(self.memory.targets)
        for i in range(1, 7):
            print(i, targets[targets == i].shape[0])

        num_keys = len(self.data.task_set) + len(self.memory.task_set)
        assert batch_size >= num_keys, "Batch size must be greator or equal "\
                                       "to the number of tasks in the memory " \
                                       "and current data."

        single_group_batch_size = batch_size // num_keys
        remaining_example = batch_size % num_keys

        self.loader_data, remaining_example = self._create_dataloaders(
            data, 4,
            remaining_example, **kwargs)
        self.loader_memory, remaining_example = self._create_dataloaders_upsample(
            memory, 6,
            remaining_example, data.targets, **kwargs)

        max_len1 = max([len(d) for d in self.loader_data.values()])
        max_len2 = max([len(d) for d in self.loader_memory.values()])

        print(max_len1, max_len2)
        if max_len1 >= max_len2:
            self.max_len = max_len1
        else:
            remain = max_len2 - max_len1
            self.max_len = max_len1 + remain // 2

    def __iter__(self):
        iter_data_dataloaders = {}
        iter_buffer_dataloaders = {}

        for t in self.loader_data.keys():
            iter_data_dataloaders[t] = iter(self.loader_data[t])
        for t in self.loader_memory.keys():
            iter_buffer_dataloaders[t] = iter(self.loader_memory[t])

        max_len = max([len(d) for d in chain(iter_data_dataloaders.values(),
                                             iter_buffer_dataloaders.values())])
        try:
            for it in range(self.max_len):
                mb_curr = []
                while len(mb_curr) != 2:
                    self._get_mini_batch_from_data_dict(
                        self.data, iter_data_dataloaders,
                        self.loader_data, self.oversample_small_tasks,
                        mb_curr)

                    self._get_mini_batch_from_data_dict(
                        self.memory, iter_buffer_dataloaders,
                        self.loader_memory, self.oversample_small_tasks,
                        mb_curr)

                yield self.collate_mbatches(mb_curr)
        except StopIteration:
            return

    def __len__(self):
        return self.max_len

    def _get_mini_batch_from_data_dict(self, data, iter_dataloaders,
                                       loaders_dict, oversample_small_tasks,
                                       mb_curr):
        # list() is necessary because we may remove keys from the
        # dictionary. This would break the generator.
        for t in list(iter_dataloaders.keys()):
            t_loader = iter_dataloaders[t]
            try:
                tbatch = next(t_loader)
            except StopIteration:
                # StopIteration is thrown if dataset ends.
                # reinitialize data loader
                if oversample_small_tasks:
                    # reinitialize data loader
                    iter_dataloaders[t] = iter(loaders_dict[t])
                    tbatch = next(iter_dataloaders[t])
                else:
                    del iter_dataloaders[t]
                    continue
            mb_curr.append(tbatch)

    def _create_dataloaders(self, data_dict, single_exp_batch_size,
                            remaining_example, **kwargs):
        loaders_dict: Dict[int, DataLoader] = {}
        for task_id in data_dict.task_set:
            data = data_dict.task_set[task_id]
            current_batch_size = single_exp_batch_size
            if remaining_example > 0:
                current_batch_size += 1
                remaining_example -= 1
            loaders_dict[task_id] = DataLoader(
                data, batch_size=current_batch_size, **kwargs)
        return loaders_dict, remaining_example

    def _create_dataloaders_upsample(self, data_dict, single_exp_batch_size,
                            remaining_example, targets, **kwargs):
        targets = np.array(targets)
        max_num = max([targets[targets == i].shape[0] for i in range(1, 7)])
        loaders_dict: Dict[int, DataLoader] = {}
        for task_id in data_dict.task_set:
            data = data_dict.task_set[task_id]
            current_batch_size = single_exp_batch_size
            if remaining_example > 0:
                current_batch_size += 1
                remaining_example -= 1

            # sampler = BalanceClassSampler(labels=data.targets, mode='upsampling')
            targets = np.array(data.targets)
            class_weights = []
            for i in range(7):
                data_num = (targets == i).shape[0] 
                if data_num != 0:
                    class_weights.append(max_num / data_num)
                else:
                    class_weights.append(0)

            weights = np.zeros(targets.shape[0])
            for i in range(1, 7):
                weights[targets == i] = class_weights[i]

            sampler = WeightedRandomSampler(weights, 3 * max_num, replacement=True)
            loaders_dict[task_id] = DataLoader(
                data, batch_size=current_batch_size, sampler=sampler, **kwargs)
        return loaders_dict, remaining_example


__all__ = [
    'TaskBalancedDataLoader',
    'GroupBalancedDataLoader',
    'ReplayDataLoader'
]
