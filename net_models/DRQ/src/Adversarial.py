import torch
import torch.nn.functional as F
import torch.optim
from .CustomEnums import DataSetName, AdversarialAttacks, Norm, LossFunction
from .Losses import get_loss
from .Datasets import get_lower_and_upper_limits
from src.autoattack.autopgd_pt import APGDAttack
from src.autoattack.square import SquareAttack
from src.autoattack.fab_pt import FABAttack_PT
from src.Datasets import get_dataset_information
import kornia
from torchvision import transforms

def get_eps(conf):
    if conf.dataset == DataSetName.cifar10 or conf.dataset == DataSetName.cifar10c or conf.dataset == DataSetName.cifar100 or conf.dataset == DataSetName.cifar100c or conf.dataset == DataSetName.svhn:
        eps = 8. / 255.
    else:
        raise NameError("No perturbation budget defined for this dataset")
    return eps

def get_alpha(epsilon):
    return epsilon / 4

def get_delta(eps, X, uniform, gradient=False):
    delta = torch.zeros(X.shape, device="cuda")
    if uniform:
        delta.uniform_(-eps, eps)
    lower_limit, upper_limit = get_lower_and_upper_limits()
    delta = clamp(delta, lower_limit - X.detach(), upper_limit - X.detach())
    if gradient:
        delta.requires_grad = True
    return delta

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def get_adversarial_attack(conf, model, attack_type, X, y, attack_conf):
    delta = torch.zeros_like(X).cuda()
    if attack_type == AdversarialAttacks.pgd:
        delta = pgd(conf, model, X, y, attack_conf=attack_conf)
    elif attack_type == AdversarialAttacks.fgsm:
        delta = fgsm(conf, model, X, y, attack_conf)
    elif attack_type == AdversarialAttacks.noise_uniform:
        delta = noise_uniform(conf, X, attack_conf)
    elif attack_type == AdversarialAttacks.noise_gauss:
        delta = gauss_noise(conf, X, attack_conf)
    elif attack_type == AdversarialAttacks.apgd:
        delta = do_autoattack(conf, model, X, y, attack_conf)
    elif attack_type == AdversarialAttacks.rotation:
        return adversarial_rotation(model, X, y, attack_conf)
    elif attack_type == AdversarialAttacks.fab:
        return do_fab(conf, model, X, y, attack_conf)
    elif attack_type == AdversarialAttacks.square:
        return do_square(conf, model, X, y, attack_conf)
    elif attack_type == AdversarialAttacks.translation:
        return adversarial_translation(model, X, y, conf, attack_conf)

    return delta

# get noise augmentation. Magnitude is dependent on noise level
def noise_uniform(conf, X, attack_conf=None):
    nl = get_value("noise", 1, attack_conf)
    eps = nl * get_eps(conf)
    delta = get_delta(eps, X, True)
    return delta

# get gauss noise augmentation. Magnitude is dependent on noise level
def gauss_noise(conf, X, attack_conf=None):
    nl = get_value("noise", 1, attack_conf)
    delta = torch.randn_like(X) * nl * get_eps(conf)
    return delta

def rot_img(x, theta, dtype=torch.cuda.FloatTensor):
    def get_rot_mat(theta):
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x

def translate_img(x, t, size):
    transform_list = []
    transform_list.append(transforms.Pad(t, padding_mode='constant'))
    transform_list.append(transforms.RandomCrop(size[2:], padding=t))
    transform = transforms.Compose(transform_list)
    x_t = transform(x)
    return x_t

def adversarial_rotation(model, X, y, attack_conf=None):
    rot = get_value("rotation", 30, attack_conf)
    angles = torch.arange(-rot, rot, 1, dtype=torch.float, device='cuda')
    #X = X.cpu()
    best_angles = torch.zeros(X.shape[0], device='cuda')
    delta = torch.zeros_like(X)
    for angle in angles:
        rotate = kornia.rotate(X, angle)
        #rotate = rot_img(X, angle)
        d = rotate - X
        pred = model((X + d).cuda())
        index = torch.where(pred.max(1)[1] != y)[0]
        delta.data[index] = d[index]
        best_angles[index] = angle
    return delta.cuda()

def adversarial_translation(model, X, y, conf, attack_conf=None):
    translation = get_value("translation", 5, attack_conf)
    tx = (torch.rand(X.shape[0]) * 2 * translation).type(torch.long) - translation
    ty = (torch.rand(X.shape[0]) * 2 * translation).type(torch.long) - translation
    size = get_dataset_information(conf.dataset)["shape"]
    delta = torch.zeros_like(X)
    translated = translate_img(X, translation, size)
    d = translated - X
    pred = model((X + d).cuda())
    delta.data = d
    #index = torch.where(pred.max(1)[1] != y)[0]
    #delta.data[index] = d[index]
    return delta.cuda()

# fgsm attack
def fgsm(conf, model, X, y, attack_conf):
    loss_function = get_value("loss", LossFunction.sce, attack_conf)
    if "valid" in loss_function.name:
        raise Exception("VALID loss not defined for FGSM")
    epsilon = get_eps(conf)
    lower_limit, upper_limit = get_lower_and_upper_limits()
    delta = get_delta(epsilon, X, False)
    delta.requires_grad = True
    pred = model(X + delta)
    loss = get_loss(conf, loss_function, X, X, pred, pred, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = delta + epsilon * torch.sign(grad)
    delta.data = torch.max(torch.min(upper_limit - X, delta.data), lower_limit - X)
    return delta.detach()

# pgd attack where the attack is not updated for samples where it was already successful
# (this gives a better lower bound on the robustness)
def pgd(conf, model, X, y, attack_conf=None):
    lower_limit, upper_limit = get_lower_and_upper_limits()
    epsilon = get_value('eps', get_eps(conf), attack_conf)
    restarts = get_value("restarts", 1, attack_conf)
    attack_iters = get_value("iters", 7, attack_conf)
    alpha = get_alpha(epsilon) * get_value("alpha_mult", 1, attack_conf)
    norm = get_value("norm", Norm.linf, attack_conf)
    loss_function = get_value("loss", LossFunction.ce, attack_conf)
    project = get_value("project", True, attack_conf)
    early_stopping = get_value("early_stopping", True, attack_conf)
    quantasize = get_value("quantasize", False, attack_conf)

    delta = get_delta(epsilon, X, True)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    # Restarting the attack to prevent getting stuck
    for i in range(restarts):
        delta.requires_grad = True
        if i > 0:
            pred = model(X + delta)
            index = torch.where(pred.max(1)[1] == y)[0]
            delta.data[index] = get_delta(epsilon, X, True)[index]

        for _ in range(attack_iters):
            X_adv = X + delta

            if quantasize:
                X_adv = torch.round(X_adv / quantasize) * quantasize

            pred = model(X_adv)
            # indexes are used to determine which samples needs to be updated
            if early_stopping:
                index = torch.where(pred.max(1)[1] == y)[0]
            else:
                index = torch.arange(0, len(pred), dtype=torch.long).cuda()
            if len(index) == 0:
                return delta.detach()

            grad = EOT(conf, model, loss_function, epsilon, alpha, X, delta, y, norm, attack_conf)

            if norm == Norm.linf:
                d = delta + alpha * torch.sign(grad)
            elif norm == Norm.l2:
                d = delta + alpha / torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(grad.shape[0], 1, 1, 1)

            if project:
                if norm == Norm.linf:
                    d = torch.clamp(d, -epsilon, epsilon)
                elif norm == Norm.l2:
                    d_norm = torch.norm(d.view(d.shape[0], -1), p=2, dim=1).view(d.shape[0], 1, 1, 1).squeeze()
                    mask = d_norm > epsilon
                    if torch.any(mask):
                        d[mask] = (d / d_norm.view(-1, 1, 1, 1) * epsilon)[mask]
            d = clamp(d, lower_limit - X, upper_limit - X)
            delta.data[index] = d[index]
    return delta.detach()

def EOT(conf, model, loss_function, eps, alpha, X, delta, y, norm, attack_conf):
    grad = torch.zeros_like(X)
    eot_type = get_value("EOT_type", "None", attack_conf)
    eot_iters = get_value("EOT_iters", 1, attack_conf)
    eot_attack_iters = get_value("EOT_attack_iters", 7, attack_conf)
    eps_factor = get_value("EOT_eps_factor", 1, attack_conf)
    delta_t = get_delta(eps * eps_factor, X + delta, True, True)
    for _ in range(eot_iters):
        with torch.enable_grad():
            if eot_type == "Attack":
                for i in range(eot_attack_iters):
                    pred = model(X + delta + delta_t)
                    loss = get_loss(conf, loss_function, X.data, delta, pred, y, attack_conf)
                    loss.backward()
                    grad = delta_t.grad.detach()
                    if norm == Norm.linf:
                        delta_t.data = delta_t + alpha * torch.sign(grad)
                        delta_t.data = torch.clamp(delta_t, -eps * eps_factor, eps * eps_factor)
                    elif norm == Norm.l2:
                        delta_t.data = delta_t + alpha / torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(grad.shape[0], 1, 1, 1)
                        d_norm = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1).view(delta.shape[0], 1, 1, 1).squeeze()
                        mask = d_norm > eps * eps_factor
                        if torch.any(mask):
                            delta_t.data[mask] = (delta_t / d_norm.view(-1, 1, 1, 1) * eps * eps_factor)[mask]
                    delta_t.data = clamp(delta_t, 0. - X, 1. - X)
                    delta_t.grad.zero_()
            if eot_type == "Noise":
                delta_t = get_delta(eps * eps_factor, X, True, True)
        pred = model(X + delta + delta_t)
        loss = get_loss(conf, loss_function, X.data, delta + delta_t, pred, y, attack_conf)
        grad += torch.autograd.grad(loss, [delta])[0].detach()

    grad /= float(eot_iters)
    return grad

def do_autoattack(conf, model, X, y, attack_conf):
    attack_iters = get_value("iters", 100, attack_conf)
    restarts = get_value("restarts", 1, attack_conf)
    norm = get_value("norm", Norm.linf, attack_conf)
    eps = get_value("eps", get_eps(conf), attack_conf)
    loss = get_value("loss", "ce", attack_conf)
    eot = get_value("EOT", 1, attack_conf)
    if isinstance(loss, LossFunction):
        loss = loss.name

    if norm == Norm.linf:
        norm = "Linf"
    elif norm == Norm.l2:
        norm = "L2"

    attack = APGDAttack(model, n_iter=attack_iters, n_restarts=restarts, norm=norm, eps=eps, loss=loss, eot_iter=eot, eps_inf = get_eps(conf))
    _, x_adv = attack.perturb(X, y, cheap=True)
    delta = x_adv - X
    return delta

def do_square(conf, model, X, y, attack_conf):
    attack_iters = get_value("iters", 5000, attack_conf)
    norm = get_value("norm", Norm.linf, attack_conf)
    eps = get_value("eps", get_eps(conf), attack_conf)

    if norm == Norm.linf:
        norm = "Linf"
    elif norm == Norm.l2:
        norm = "L2"

    attack = SquareAttack(model, n_queries=attack_iters, norm=norm, eps=eps)
    x_adv = attack.perturb(X, y)
    delta = x_adv - X
    return delta

def do_fab(conf, model, X, y, attack_conf):
    attack_iters = get_value("iters", 100, attack_conf)
    norm = get_value("norm", Norm.linf, attack_conf)
    eps = get_value("eps", get_eps(conf), attack_conf)

    if norm == Norm.linf:
        norm = "Linf"
    elif norm == Norm.l2:
        norm = "L2"

    attack = FABAttack_PT(model, n_iter=attack_iters, norm=norm, eps=eps)
    x_adv = attack.perturb(X, y)
    delta = x_adv - X
    return delta

def get_value(key, value, conf):
    if key in conf:
        return conf[key]
    else:
        return value