


def get_num_of_classes(args):
    if args.dataset in ['cifar10', 'cifar10-c']:
        num_classes = 10
    elif args.dataset in ['cifar100', 'cifar100-c']:
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000

    return num_classes
