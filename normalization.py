
def get_normalization_stats(args):
    if args.dataset in ['cifar10', 'cifar10-c']:
        if args.net_name in ['at', 'pat']:
            return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if args.net_name in ['rebuffi', 'gowal']:
            return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        if args.net_name in ['clean']:
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    if args.dataset in ['cifar100', 'cifar100-c']:
        if args.net_name == 'at':
            raise Exception('no normalization')
        if args.net_name in ['rebuffi', 'gowal']:
            return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)


    if args.dataset in ['imagenet']:
        if args.net_name in ['at', 'do']:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

