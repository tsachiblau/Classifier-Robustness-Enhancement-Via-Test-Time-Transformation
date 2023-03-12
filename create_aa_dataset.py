import os
import numpy as np
from get_dataset import get_data_loader
from autoattack import AutoAttack
import torch

def create_aa_dataset(model, args):
    batch_size = args.batch_size
    if args.dataset in ['cifar10', 'cifar100']:
        args.batch_size = 10000
    elif args.dataset in ['imagenet']:
        args.batch_size = 50000
    testloader = get_data_loader(args)
    args.batch_size = batch_size

    model.eval()
    x, y = next(iter(testloader))
    norm = args.attack_threat_model
    eps = args.attack_epsilon
    print('norm {},   epsilon {}'.format(norm, eps))

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=True)
    x_adv = adversary.run_standard_evaluation(x, y, bs=args.batch_size)

    if args.defense_method not in ['TTE']:
        if not os.path.exists(os.path.dirname(args.aa_dataset_path)):
            os.makedirs(os.path.dirname(args.aa_dataset_path), exist_ok=True)
        if not os.path.exists(os.path.dirname(args.aa_labels_path)):
            os.makedirs(os.path.dirname(args.aa_labels_path), exist_ok=True)

        torch.save(x_adv, args.aa_dataset_path)
        torch.save(y, args.aa_labels_path)
