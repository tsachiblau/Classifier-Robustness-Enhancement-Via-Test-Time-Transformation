import torch
import itertools


from net_models.DRQ.read_yaml import open_yaml
from attacks import get_defense_threat_model
from get_num_of_classes import get_num_of_classes
from net_models.DRQ.src.DecisionRegionQuantification import decision_region_robustness
from net_models.DRQ.src.Configuration import Conf
from analysis import plot_figure
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def add_defense(model, args):
    if args.defense_method in ['TETRA']:
        return TETRA(model, args)
    elif args.defense_method in ['DRQ']:
        return DRQ(model, args)
    elif args.defense_method in ['TTE']:
        return TTE(model, args)
    elif args.defense_method in ['None']:
        return Defense(model, args)
    else:
        raise Exception('Unrecognized defense method')


class Defense(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.N_classes = get_num_of_classes(args)
    def forward(self, x, use_defense=False):
        if use_defense==False:
            return self.model(x)
        else:
            return self.get_defense_prediction_function(x)

    def get_defense_prediction_function(self, x):
        return self.model(x)



class DRQ(Defense):
    def __init__(self, model, args):
        super().__init__(model, args)
        yaml_file = './net_models/DRQ/yaml/cifar10.yaml'
        arguments, keys = open_yaml(yaml_file)
        runs = list(itertools.product(*arguments))
        run = runs[0]
        current_args = {}
        current_args["experiment_path"] = ''#input_args.experiment_path
        current_args["data_path"] = ''#input_args.data_path
        for i in range(len(run)):
            current_args[keys[i]] = run[i]

        self.counterattack_parameters = Conf(current_args).counterattack_defense


    def get_defense_prediction_function(self, x):
        logits = decision_region_robustness(self, x, self.N_classes, self.counterattack_parameters)
        dict_res = {}
        dict_res['logits'] = logits[0]

        return dict_res



class TTE(Defense):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.list_trans = self.get_transforms()


    def get_transforms(self):
        if self.args.dataset in ['cifar10', 'cifar100']:
            if self.args.net_name in ['at', 'rebuffi', 'gowal']:
                list_trans = []
                list_trans.append(
                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(p=0.0),
                    ])
                )
                list_trans.append(
                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    ])
                )

                for _ in range(4):
                    list_trans.append(
                        torchvision.transforms.Compose([
                        torchvision.transforms.RandomCrop(32, padding=4)
                        ])
                    )

                for _ in range(4):
                    list_trans.append(
                        torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(p=1.0),
                        torchvision.transforms.RandomCrop(32, padding=4)
                        ])
                    )
        if self.args.dataset in ['imagenet']:
            if self.args.net_name in ['do', 'at']:
                list_trans = []
                list_trans.append(
                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(p=0.0),
                    ])
                )
                list_trans.append(
                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(p=1.0),
                    ])
                )

                for _ in range(4):
                    list_trans.append(
                        torchvision.transforms.Compose([
                        torchvision.transforms.RandomCrop(224, padding=0)
                        ])
                    )

                for _ in range(4):
                    list_trans.append(
                        torchvision.transforms.Compose([
                        torchvision.transforms.RandomHorizontalFlip(p=1.0),
                        torchvision.transforms.RandomCrop(224, padding=0)
                        ])
                    )
        return list_trans


    def get_defense_prediction_function(self, x):

        list_logits = []
        for t in self.list_trans:
            trans_x = t(x)
            logits = self.model(trans_x)
            list_logits.append(logits.unsqueeze(0))

        avg_logits = torch.concat(list_logits, dim=0).mean(0)

        dict_res = {'logits': avg_logits}
        return dict_res






class TETRA(Defense):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.defense_pgd = get_defense_threat_model(args)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()

    def get_defense_prediction_function(self, x):
        repeat_y = torch.Tensor([int(i) for i in range(self.N_classes)]).reshape((-1, 1)).repeat((1, len(x))).flatten().long().cuda()
        repeat_x = x.repeat((self.N_classes, 1, 1, 1))
        ktop = self.args.speed_up_ktop

        tensor_prediction = torch.zeros((len(repeat_y), )).to(x.device)

        if self.args.use_ktop:
            with torch.no_grad():
                tensor_logits = self.model(x)
            tensor_argsort = tensor_logits.sort(dim=1, descending=True)[0]
            tendor_bool_calc = tensor_logits > (tensor_argsort[:, ktop].unsqueeze(1))
            tensor_bool_repreat = tendor_bool_calc.permute(1, 0).reshape(-1,)
        else:
            tensor_bool_repreat = torch.ones_like(repeat_y).bool()

        repeat_x, repeat_y = repeat_x[tensor_bool_repreat], repeat_y[tensor_bool_repreat]

        tensor_reverse_pgd, images_distance = self.defense_pgd.get_adv_x(self.model, repeat_x, repeat_y)

        res_dict = {}
        # set MSE distance
        tensor_prediction[tensor_bool_repreat] = 1 / (1e-10 + torch.nn.MSELoss(reduction='none')(tensor_reverse_pgd, repeat_x).sum(dim=(1, 2, 3)))
        res_dict['MSE'] = tensor_prediction.reshape(self.N_classes, -1).permute(1, 0).clone()

        dif_vec = (tensor_reverse_pgd - repeat_x).view(len(tensor_reverse_pgd), -1)
        tensor_prediction[tensor_bool_repreat] = 1 / (1e-10 + torch.norm(dif_vec, p=1, dim=1))
        res_dict['MAE'] = tensor_prediction.reshape(self.N_classes, -1).permute(1, 0).clone()

        for i in range(len(tensor_reverse_pgd)):
            tensor_prediction[tensor_bool_repreat][i] = 1 / (1e-10 + self.lpips(tensor_reverse_pgd[i].unsqueeze(0), repeat_x[i].unsqueeze(0)))
        res_dict['LPIPS'] = tensor_prediction.reshape(self.N_classes, -1).permute(1, 0).clone()

        if self.args.plot:
            plot_figure(x, tensor_reverse_pgd, repeat_y, self.args)

        return res_dict

