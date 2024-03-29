import torch
import torchvision.models
from torch.nn import Linear
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy

import argparse

from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, class_accuracy_metrics
from avalanche.logging import TextLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LwFPlugin, CWRStarPlugin, SynapticIntelligencePlugin, \
ClassBalancedStoragePolicy, AGEMPlugin, CoPEPlugin, EWCPlugin
from avalanche.training.strategies import Naive
from avalanche.models import IcarlNet, SLDAResNetModel

from class_strategy import *
from classification_util import *
from lr_scheduler import LRSchedulerIterPlugin
from losses import FocalLoss
from lfl import LFLPlugin
from replay import ReplayPlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='result',
                        help='Name of the result files')
    parser.add_argument('--root', default="/data/aza_s/codalab/SSLAD-2D",
                        help='Root folder where the data is stored')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Num workers to use for dataloading. Recommended to have more than 1')
    parser.add_argument('--store', action='store_true',
                        help="If set the prediciton files required for submission will be created")
    parser.add_argument('--test', action='store_true',
                        help='If set model will be evaluated on test set, else on validation set')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If set, training will be on the CPU')
    parser.add_argument('--store_model', action='store_true',
                        help="Stores model if specified. Has no effect is store is not set")
    args = parser.parse_args()

    ######################################
    #                                    #
    # Editing below this line allowed    #
    #                                    #
    ######################################

    args.root = f"{args.root}/labeled"
    device = torch.device('cuda:6' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = Linear(2048, 7, bias=True)

    # model = create_model(
    #         model_name='densenet201',
    #         pretrained=True,
    #         num_classes=7,
    #     )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': 0.0001}], lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1)
    
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0., 20., 10., 1., 1., 20., 300.]).to(device))
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy(0.1)
    # criterion = FocalLoss(10, 10)
    batch_size = 10

    # Add any additional plugins to be used by Avalanche to this list. A template
    # is provided in class_strategy.py.

    sch_plugin = LRSchedulerIterPlugin(scheduler)
    # replay = ReplayPlugin(1000, ClassBalancedStoragePolicy({}, 1000))
    replay = ReplayPlugin(1000)
    cwr = CWRStarPlugin(model, freeze_remaining_model=False)
    agem = AGEMPlugin(1000, 100)
    # plugins = [ClassStrategyPlugin(), cwr, replay]    
    plugins = [ClassStrategyPlugin(), replay, sch_plugin]    

    ######################################
    #                                    #
    # No editing below this line allowed #
    #                                    #
    ######################################

    if batch_size > 10:
        raise ValueError(f"Batch size {batch_size} not allowed, should be less than or equal to 10")

    img_size = 64
    train_sets = create_train_set(args.root, img_size)

    evaluate = 'test' if args.test else 'val'
    if evaluate == "val":
        test_sets = create_val_set(args.root, img_size)
    else:
        test_sets, _ = create_test_set_from_pkl(args.root, img_size)

    benchmark = create_multi_dataset_generic_benchmark(train_datasets=train_sets, test_datasets=test_sets)

    text_logger = TextLogger(open(f"./{args.name}.log", 'w'))
    interactive_logger = InteractiveLogger()
    store = args.name if args.store else None

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(stream=True), loss_metrics(stream=True),
        class_accuracy_metrics(stream=True),
        ClassEvaluationPlugin(reset_at='stream', emit_at='stream', mode='eval',
                              store=store),
        loggers=[text_logger, interactive_logger])

    strategy = Naive(
        model, optimizer, criterion, train_mb_size=batch_size, train_epochs=1, eval_mb_size=256, device=device,
        evaluator=eval_plugin, eval_every=1, plugins=plugins)

    accuracies_test = []
    for i, experience in enumerate(benchmark.train_stream):
        # Shuffle will be passed through to dataloader creator.
        strategy.train(experience, eval_streams=[], shuffle=False, num_workers=args.num_workers)

        results = strategy.eval(benchmark.test_stream, num_workers=args.num_workers)
        mean_acc = [r[1] for r in results['Top1_ClassAcc_Stream/eval_phase/test_stream/Task000']]
        print(f'Mean {sum(mean_acc) / len(mean_acc)}')
        accuracies_test.append(sum(mean_acc) / len(mean_acc))

    print(f"Average mean test accuracy: {sum(accuracies_test) / len(accuracies_test) * 100:.3f}%")
    print(f"Average mean test accuracy: {sum(accuracies_test) / len(accuracies_test) * 100:.3f}%",
          file=open(f'./{args.name}.log', 'a'))
    print(f"Final mean test accuracy: {accuracies_test[-1] * 100:.3f}%")
    print(f"Final mean test accuracy: {accuracies_test[-1] * 100:.3f}%",
          file=open(f'./{args.name}.log', 'a'))

    if args.store_model:
        torch.save(model.state_dict(), f'./{args.name}.pt')


if __name__ == '__main__':
    main()
