import os
import random
import argparse

from attacks import *
from load_model import load_model
from eval_model import eval_model
from create_aa_dataset import create_aa_dataset
from add_defense import add_defense

parser = argparse.ArgumentParser(description='ReversePGD')

## general
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='random seed')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--debug', type=bool, default=False, help='Wehther to use debug mode')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--quick_review', type=bool, default=False, help='True, False')
parser.add_argument('--plot', type=bool, default=False, help='Wehther to use debug mode')

# --defense_gamma 5500 --defense_alpha 6.0 --net_threat_model L2_3.0 --attack_info L2_6.0 --defense_method TETRA --dataset imagenet --net_name at --batch_size 1 --debug True --defense_num_steps 30 --aa_dataset True --use_ktop True --speed_up_ktop 5 --plot True
## model
parser.add_argument('--net_name', default='uncovering')
parser.add_argument('--net_arch', default='wrn')
parser.add_argument('--net_depth', type=int, default=28)
parser.add_argument('--net_width', type=int, default=10)
parser.add_argument('--net_threat_model', type=str, default='L2_0.5')

## attack
parser.add_argument('--attack_info', type=str, default='Linf_8', help='perturbation')
parser.add_argument('--attack_epsilon', type=float, default=8/255, help='perturbation')
parser.add_argument('--attack_threat_model', default='Linf', help='Linf, L2')
parser.add_argument('--attack_num_steps', default=20, help='perturb number of steps')
parser.add_argument('--attack_alpha', default=-1, help='perturb number of steps')
parser.add_argument('--aa_dataset', type=bool, default=False, help='True, False')
parser.add_argument('--aa_dataset_path', type=str, default='')
parser.add_argument('--aa_labels_path', type=str, default='')

## defense
parser.add_argument('--defense_method', type=str, default='TETRA', help='L2, Linf')
parser.add_argument('--defense_threat_model', type=str, default='L2', help='L2, Linf')
parser.add_argument('--defense_num_steps', type=int, default=100, help='perturb number of steps')
parser.add_argument('--defense_alpha', type=float, default=-1, help='perturb number of steps')
parser.add_argument('--defense_gamma', type=float, default=1e-1, help='perturb number of steps')
parser.add_argument('--speed_up_ktop', type=int, default=10, help='perturb number of steps')
parser.add_argument('--use_ktop', type=bool, default=False)
parser.add_argument('--ktop_loss', type=bool, default=False)


## create AA dataset
parser.add_argument('--flow', type=str, default='eval', help='perturb number of steps')


args = parser.parse_args()

if __name__ == '__main__':



    if args.net_name in ['clean']:
        args.net_depth = 28
        args.net_width = 10
        args.net_threat_model = 'none'

    if args.net_name in ['pat']:
        args.net_arch = 'rn'
        args.net_depth = 50
        args.net_width = 0
        args.net_threat_model = 'none'

    if args.net_threat_model == 'L2_0.5':
        if args.net_name == 'rebuffi':
            args.net_arch = 'wrn'
            args.net_depth = 28
            args.net_width = 10
        elif args.net_name == 'at':
            args.net_arch = 'rn'
            args.net_depth = 50
            args.net_width = 0
    elif args.net_threat_model in ['L2_3.0', 'Linf_4', 'Linf_8']:
        if args.net_name == 'rebuffi':
            args.net_arch = 'wrn'
            args.net_depth = 28
            args.net_width = 10
        elif args.net_name == 'gowal':
            args.net_arch = 'wrn'
            args.net_depth = 70
            args.net_width = 16
        elif args.net_name == 'at':
            args.net_arch = 'rn'
            args.net_depth = 50
            args.net_width = 0
        elif args.net_name == 'do':
            args.net_arch = 'rn'
            args.net_depth = 50
            args.net_width = 2

    if args.attack_info not in ['None']:
        attack_type, threat_model_norm = args.attack_info.split('_')
        args.attack_threat_model = attack_type

        if threat_model_norm in ['4', '8', '16']:
            args.attack_epsilon = float(threat_model_norm) / 255
        else:
            args.attack_epsilon = float(threat_model_norm)


    norm_disp = args.attack_epsilon
    if args.attack_threat_model == 'Linf':
        if np.linalg.norm(args.attack_epsilon - 8/255) < 1e-3:
            norm_disp = 8
        elif np.linalg.norm(args.attack_epsilon - 16/255) < 1e-3:
            norm_disp = 16
        elif np.linalg.norm(args.attack_epsilon - 4 / 255) < 1e-3:
            norm_disp = 4
        else:
            raise Exception('Unrecognized attack threat model')


    args.aa_dataset_path = os.path.join('data', args.net_name, args.dataset, args.net_threat_model, args.attack_threat_model + '_' + str(norm_disp) + '_attack.pt')
    args.aa_labels_path = os.path.join('data', args.net_name, args.dataset, args.net_threat_model, args.attack_threat_model + '_' + str(norm_disp) + '_attack_labels.pt')


    if args.aa_dataset:
        args.num_workers = 0

    args.attack_alpha = 2.5 * (args.attack_epsilon / args.attack_num_steps)

    print('#' * 100)
    print(args)
    print('#' * 100)

    os.environ['WANDB_MODE'] = 'offline' if args.debug else 'online'
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "10000"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = load_model(args)

    if args.plot:
        args.batch_size = 1

    if args.flow in ['eval']:
        model = add_defense(model, args)
        eval_model(model, args)
    elif args.flow in ['create_aa']:
        if args.defense_method in ['TTE']:
            model = add_defense(model, args)
        create_aa_dataset(model, args)
