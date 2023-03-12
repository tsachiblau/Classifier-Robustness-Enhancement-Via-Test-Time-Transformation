import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import torchsort
from get_num_of_classes import get_num_of_classes


def get_threat_model(args):
    if args.attack_info == 'None' or args.aa_dataset:
        attack = no_attack()
    else:
        if args.ktop_loss:
            attack = pgd_L2_ranking(args.attack_epsilon, args.attack_num_steps, args.attack_alpha, args)
        else:
            if args.attack_threat_model =='Linf':
                attack = pgd_linf(args.attack_epsilon, args.attack_num_steps, args.attack_alpha)
            elif args.attack_threat_model =='L2':
                attack = pgd_l2(args.attack_epsilon, args.attack_num_steps, args.attack_alpha)



    return attack

def get_defense_threat_model(args):
    if args.defense_threat_model in ['L2']:
        return defense_targeted_pgd_l2(args.defense_num_steps, args.defense_alpha, args.defense_gamma)
    # elif args.defense_threat_model in ['Linf']:
    #     return defense_targeted_pgd_linf(args.defense_epsilon, args.defense_num_steps, args.defense_alpha, args.defense_gamma)
    else:
        raise Exception('cant find defense PGD')

def ranking_loss(logits, y, num_classes, top_k):
    ranking = torchsort.soft_rank(-logits, regularization_strength=3.0)
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).bool()
    loss = (ranking[y_one_hot]).mean()

    return loss


class pgd_L2_ranking():
    def __init__(self, epsilon, num_steps, alpha, args):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.args = args
        self.num_classes = get_num_of_classes(args)
    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = ranking_loss(model(x + delta), y, self.num_classes, self.args.speed_up_ktop)
            loss.backward()
            norm = self.norms(delta.grad.detach())
            bool_norm = (norm != 0).reshape(-1,)
            delta.data[bool_norm] += self.alpha * delta.grad.detach()[bool_norm] / norm[bool_norm]
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.grad.zero_()
        return x + delta.detach()


class pgd_l2():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward()
            delta.data += self.alpha * delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.grad.zero_()
        return x + delta.detach()



class no_attack():
    def get_adv_x(self, model, x, y):
        return x

class pgd_linf():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha

    def get_adv_x(self, model, x, y):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward()
            delta.data = (delta.data + self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.data = (x + delta.data).clamp(0, 1) - x
        return x + delta.detach()


class targeted_pgd_linf():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha

    def get_adv_x(self, model, x, y_target):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss.backward()
            delta.data = (delta.data - self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.data = (x + delta.data).clamp(0, 1) - x
        return x + delta.detach()




class targeted_pgd_l2():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha

    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y_target):
        delta = torch.zeros_like(x, requires_grad=True)
        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss.backward()
            delta.data -= self.alpha * delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.data *= self.epsilon / self.norms(delta.detach()).clamp(min=self.epsilon)
            delta.grad.zero_()

        return x + delta.detach()


########################################################################################################################
########################################################################################################################
################################################## Defense #############################################################
########################################################################################################################
########################################################################################################################


class defense_targeted_pgd_linf():
    def __init__(self, epsilon, num_steps, alpha, gamma, defense_distance):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma
        self.defense_distance = defense_distance



    def get_adv_x(self, model, x, y_target):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss1 = nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss2 = self.gamma * nn.MSELoss()(delta, torch.zeros_like(delta))
            loss = loss1 + loss2
            loss.backward()
            delta.data = (delta.data - self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.data = (x + delta.data).clamp(0, 1) - x
        return x + delta.detach()


'''
class defense_targeted_pgd_l2_tmp():
    def __init__(self, num_steps, alpha, gamma):
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma
        self.distance_fn = torch.nn.MSELoss()
        # self.distance_fn = lpips.LPIPS(net='alex', verbose=False).cuda()

    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y_target):
        th = 1e-3
        delta = torch.zeros_like(x, requires_grad=True)
        counter_steps = torch.zeros((len(x))).cuda()
        images_distance = torch.zeros((len(x))).cuda()

        for t in range(self.num_steps):
            loss1 = self.alpha * nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss2 = self.gamma * self.distance_fn(x, x + delta).mean()
            loss = loss1 + loss2
            loss.backward()
            norm_i = self.norms(delta.grad.detach())
            counter_steps += norm_i.resize(len(x)) > th
            images_distance += norm_i.resize(len(x))

            delta.data -= self.alpha * delta.grad.detach() / norm_i
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.grad.zero_()
            if t < self.num_steps - 1:
                delta.data += torch.randn_like(delta) * 0.001

        return x + delta.detach(), images_distance
'''


class defense_targeted_pgd_l2():
    def __init__(self, num_steps, alpha, gamma):
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma
        self.distance_fn = torch.nn.MSELoss()


    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y_target):
        instance_dist = torch.zeros(len(x),).cuda()
        delta = torch.zeros_like(x, requires_grad=True)
        for t in range(self.num_steps):
            loss1 = nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss2 = self.gamma * self.distance_fn(x, x + delta).mean()
            loss = loss1 + loss2
            loss.backward()
            norm = self.norms(delta.grad.detach())
            instance_dist += norm.view(len(x),)
            delta.data -= self.alpha * delta.grad.detach() / norm
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.grad.zero_()

            # print('iter: {}, loss: {}, loss1: {}, loss2 {},  loss diff {}'.format(t, loss.item(), loss1.item(), loss2.item()))


        return x + delta.detach(), instance_dist