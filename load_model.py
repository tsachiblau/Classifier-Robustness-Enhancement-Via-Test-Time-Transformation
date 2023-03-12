import os
import torch

from net_models.model_zoo import WideResNet
from net_models.trades import WideResNetTRADES
from net_models.at_cifar import ResNet50 as cifar_resnet50
from net_models.at_cifar import resnet50_imagenet, resnet50_2_imagenet
from net_models.wrn_28_10 import WideResNet as clean_wrn
from net_models.pat.per import CifarResNetFeatureModel, AttackerModel

from model_wrapper import modelWrapper
from normalization import get_normalization_stats
from get_num_of_classes import get_num_of_classes

def load_model(args):
    num_classes = get_num_of_classes(args)
    MEAN, STD = get_normalization_stats(args)
    net_arch = args.net_arch + str(args.net_depth)
    net_arch += '' if args.net_width == 0 else '_' + str(args.net_width)

    dataset = args.dataset
    if args.dataset in ['cifar10-c', 'cifar100-c']:
        if args.dataset in ['cifar10-c']:
            dataset = 'cifar10'
        elif args.dataset in ['cifar100-c']:
            dataset = 'cifar100'

    model_path = os.path.join('models', args.net_name, '{}_{}_{}.pt'.format(dataset, args.net_threat_model, net_arch))
    if args.net_name == 'at':
        m = cifar_resnet50(MEAN, STD, num_classes)
        if args.dataset == 'imagenet':
            m = resnet50_imagenet(MEAN, STD)

    elif args.net_name in ['rebuffi', 'gowal']:
        m = WideResNet(depth=args.net_depth, width=args.net_width, mean=MEAN, std=STD, num_classes=num_classes)
    elif args.net_name in ['clean']:
        m = clean_wrn(depth=args.net_depth, width=args.net_width, mean=MEAN, std=STD, num_classes=num_classes)
    elif args.net_name in ['pat']:
        m = cifar_resnet50(MEAN, STD, num_classes)
        states = torch.load(model_path)
        m.load_state_dict(states['model'])
        m = AttackerModel(m)
        m = CifarResNetFeatureModel(m)
        m = torch.nn.DataParallel(m)
        m_w = modelWrapper(m)
        return m_w.cuda().eval()
    elif args.net_name in ['do']:
        m = resnet50_2_imagenet(MEAN, STD)
    else:
        print('We dont know this model')
        pass

    m.load_state_dict(torch.load(model_path))
    m = torch.nn.DataParallel(m)
    m_w = modelWrapper(m)

    return m_w.cuda().eval()