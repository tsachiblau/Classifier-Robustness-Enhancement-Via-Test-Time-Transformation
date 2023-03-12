import torch
import torch.nn.functional as F
from net_models.DRQ.src.Utils import string_fraction_to_float
from net_models.DRQ.src.Losses import dlr_loss_targeted
from net_models.DRQ.src.FMN import fmn
from foolbox import Misclassification
import foolbox

def decision_region_robustness(model_class, X, classes, counterattack_parameters):
    X_c = X.clone()
    B = X_c.shape[0]

    iters_start, iters_counter, iters_eot, counter_type, norm, original_pred, pred_final, y, \
    delta_final, highest_distance, distances, epsilon_start, \
    epsilon_counter, epsilon_eot, noise_type, attack_init, step_size_start, step_size_counter, step_1, confidence, q_confidence, only_adversaries, class_idxs = init(X_c, B, classes, model_class, counterattack_parameters)

    # check decision region of top n classes or all classes
    for i in range(class_idxs.shape[1]):
        c = class_idxs[:, i]

        if step_1 == "targeted":
            delta_attack = exploration(model_class, X_c, y, c, epsilon_start, step_size_start, iters_start, norm, iters_eot, epsilon_eot)
        elif step_1 == "random":
            delta_attack = exploration_random(X_c, epsilon_start, noise_type)
        else:
            raise Exception("Search type ->", step_1, "<- Not defined")

        pred_attack = model_class.model(X + delta_attack).detach()
        y_attack = pred_attack.argmax(1).type(torch.long)
        search_mask = y_attack == c

        if counter_type == "max_loss":
            distances = quantification(model_class, X_c, y_attack, delta_attack, distances, c, epsilon_counter,
                                 step_size_counter, iters_counter, norm, search_mask, only_adversaries)
        elif counter_type == "random":
            distances = quantification_random(model_class, X_c, y_attack, delta_attack, distances, epsilon_counter, iters_counter, noise_type)

        current_distances_mask = torch.eye(classes, dtype=torch.bool, device='cuda')[c]
        higher_mask = distances[current_distances_mask] > highest_distance
        highest_distance[higher_mask] = distances[current_distances_mask][higher_mask]
        pred_final[higher_mask] = pred_attack[higher_mask]
        delta_final[higher_mask] = delta_attack[higher_mask]

    return pred_final, delta_final, distances, original_pred, epsilon_start

def init(X_c, B, classes, model_class, counterattack_parameters):
    if counterattack_parameters is None:
        counterattack_parameters = model_class.counterattack_parameters

    counter_type = counterattack_parameters["counter_type"]
    if counter_type not in ["max_loss", "random", "robustness"]:
        raise Exception("counter_type:", counter_type, "does not exist")

    def set_value(params, key, default):
        if key in params:
            return params[key]
        else:
            return default

    iters_start = set_value(counterattack_parameters, "iters_start", 0)
    iters_counter = set_value(counterattack_parameters, "iters_counter", 0)
    iters_eot = set_value(counterattack_parameters, "eot_iters", 0)

    norm = set_value(counterattack_parameters, "norm", None)

    with torch.no_grad():
        original_pred = model_class.model(X_c)
        y = original_pred.argmax(1)

    pred_final = torch.zeros((B, classes), device="cuda")
    delta_final = torch.zeros((B, *X_c.shape[1:]), device="cuda")
    highest_distance = torch.zeros((B), device='cuda') - float("inf")

    distances = torch.ones((B, classes), device="cuda") * float("inf")
    if counter_type == "robustness":
        distances = -distances

    epsilon_start = set_value(counterattack_parameters, "epsilon_start", 0)
    epsilon_start = string_fraction_to_float(epsilon_start)
    epsilon_counter = set_value(counterattack_parameters, "epsilon_counter", 0)
    epsilon_counter = string_fraction_to_float(epsilon_counter)
    epsilon_eot = set_value(counterattack_parameters, "eot_magnitude", 0)
    epsilon_eot = string_fraction_to_float(epsilon_eot)

    noise_type = "uniform"
    if "noise_type" in counterattack_parameters:
        noise_type = counterattack_parameters["noise_type"]

    if "attack_init" in counterattack_parameters:
        attack_init = counterattack_parameters["attack_init"]
    else:
        attack_init = True

    if "epsilon_finder" in counterattack_parameters:
        epsilon_finder = model_class.counterattack_parameters["epsilon_finder"]
    else:
        epsilon_finder = "None"

    if "only_adversaries" in counterattack_parameters:
        only_adversaries = counterattack_parameters["only_adversaries"]
    else:
        only_adversaries = False

    if "confidence" in counterattack_parameters:
        confidence = counterattack_parameters["confidence"]
    else:
        confidence = None

    if "q_confidence" in counterattack_parameters:
        q_confidence = counterattack_parameters["q_confidence"]
    else:
        q_confidence = None

    if epsilon_finder == "fmn":
        epsilon_start = initial_robustness(model_class, X_c, original_pred, confidence, 100, norm)
        epsilon_counter = epsilon_start / 2

    if "step_size" in counterattack_parameters:
        step_size_start = model_class.counterattack_parameters["step_size"]
        step_size_start = string_fraction_to_float(step_size_start)
    else:
        step_size_start = epsilon_start / 4

    if "step_size_counter" in counterattack_parameters:
        step_size_counter = model_class.counterattack_parameters["step_size_counter"]
        step_size_counter = string_fraction_to_float(step_size_counter)
    else:
        step_size_counter = epsilon_counter / 4

    if "step_1" in counterattack_parameters:
        step_1 = counterattack_parameters["step_1"]
    else:
        step_1 = "targeted"

    if 'top_n' in model_class.counterattack_parameters:
        top_n = model_class.counterattack_parameters["top_n"]
        sorted_class_idxs = torch.argsort(original_pred, 1, descending=True)
        class_idxs = sorted_class_idxs[:, :top_n]
        if counter_type == "max_loss":
            not_class_idxs = sorted_class_idxs[:, top_n:]
            mask = torch.eye(classes, device='cuda')[not_class_idxs].sum(1).type(torch.bool)
            distances[mask] -= float('inf')
    else:
        class_idxs = torch.argsort(original_pred, 1, descending=True)
    return iters_start, iters_counter, iters_eot, counter_type, norm, original_pred, pred_final, y, \
           delta_final, highest_distance, distances, epsilon_start, \
           epsilon_counter, epsilon_eot, noise_type, attack_init, step_size_start, step_size_counter, step_1, confidence, q_confidence, only_adversaries, class_idxs

def initial_robustness(model_class, X, pred, confidence, iters, norm_type):
    torch.cuda.empty_cache()
    B = X.shape[0]
    pred_confidence = torch.softmax(pred, 1).max(1).values
    y = pred.argmax(1)
    if confidence == "auto":
        confidence = pred_confidence

    if norm_type == "linf":
        attack = fmn.LInfFMNAttack(steps=iters, max_stepsize=1.0, confidence=confidence)
    else:
        attack = fmn.L2FMNAttack(steps=iters, max_stepsize=1.0, confidence=confidence)

    fmodel = foolbox.PyTorchModel(model_class, bounds=(0, 1))

    x_adv = attack.run(fmodel, X, Misclassification(y))
    perturbation = x_adv - X
    if norm_type == "linf":
        norm = torch.norm((perturbation).view(B, -1), p=float("inf"), dim=1).view(B, 1, 1, 1)
    else:
        norm = torch.norm((perturbation).view(B, -1), p=2, dim=1).view(B, 1, 1, 1)

    torch.cuda.empty_cache()
    return norm

def exploration(model_class, X, y, c, epsilon, step_size, iters, norm, iters_eot, epsilon_eot):
    # targeted
    delta_attack = initialize(X, epsilon, norm, gradient=True)
    for _ in range(iters):
        grad = 0
        for i in range(iters_eot + 1):
            noise = torch.rand_like(X) * epsilon_eot
            pred = model_class.model(X + delta_attack + noise)
            correct_class_mask = pred.argmax(1) == c
            loss = dlr_loss_targeted(pred, y, torch.ones((X.shape[0]), device="cuda", dtype=torch.long) * c)
            loss[correct_class_mask] = F.cross_entropy(pred, torch.ones((X.shape[0]), device="cuda", dtype=torch.long) * c, reduction='none')[correct_class_mask]
            loss = loss.sum()
            grad = grad + torch.autograd.grad(loss, delta_attack)[0]

        d = update(delta_attack, grad, -step_size, norm)
        d = project(X, d, epsilon, norm)
        d = torch.max(torch.min(1. - X, d), 0. - X)
        delta_attack.data = d
    return delta_attack

def exploration_random(X, epsilon, noise_type):
    if noise_type == "uniform":
        noise = torch.randn(X.shape, device='cuda') * epsilon
    elif noise_type == "gaussian":
        noise = torch.randn(X.shape, device='cuda') * epsilon
    else:
        raise Exception("Noise type not defined:", noise_type)
    return noise

def quantification(model_class, X, y_attack, delta_attack, distances, c, epsilon, step_size, iters, norm, search_mask, only_adversaries):
    delta_counter = torch.zeros(X.shape, device="cuda")
    delta_counter.requires_grad = True
    idxs = torch.eye(distances.shape[-1], dtype=torch.bool, device='cuda')[c]
    for _ in range(iters):
        pred_counter = model_class.model(X + delta_attack + delta_counter)
        index_counter = pred_counter.max(1)[1] == y_attack
        loss = F.cross_entropy(pred_counter, y_attack, reduction="none")
        grad = torch.autograd.grad(loss.sum(), delta_counter)[0]
        d = update(delta_counter, grad, step_size, norm)
        d = project(X, d, epsilon, norm)
        d = torch.max(torch.min(1. - (X + delta_attack), d), 0. - (X + delta_attack))
        delta_counter.data[index_counter] = d[index_counter]
        mask = (distances[idxs] > -loss)
        if only_adversaries:
            mask *= search_mask
        distances[mask.view(-1, 1) * idxs] = -loss[mask].data

    if only_adversaries and (~search_mask).sum() > 0:
        distances[~search_mask.view(-1, 1) * idxs] = -float("inf")

    return distances

def quantification_random(model_class, X, y_attack, delta_attack, distances, epsilon, iters, noise_type):
    noise = exploration_random(X, epsilon, noise_type)
    for _ in range(iters):
        pred_counter = model_class.model(X + delta_attack + noise)
        loss = F.cross_entropy(pred_counter, y_attack, reduction="none")
        index_counter = pred_counter.max(1)[1] == y_attack
        noise.data[index_counter] = exploration_random(X, epsilon, noise_type)[index_counter]
        idxs = torch.eye(pred_counter.shape[-1], dtype=torch.bool, device='cuda')[y_attack]
        mask = distances[idxs] > -loss
        distances[mask.view(-1, 1) * idxs] = -loss[mask].data
    return distances

def initialize(X, eps, norm, gradient=False, rnd=False):
    delta = torch.zeros(X.shape, device="cuda")
    if norm == "linf":
        if rnd:
            delta = delta.uniform_(-eps, eps)
    elif norm == "l2":
        if rnd:
            delta = torch.randn(X.shape, device="cuda")
            delta = torch.rand(X.shape[0], device="cuda").view(-1, 1, 1, 1) * delta.renorm(p=2, dim=0, maxnorm=eps)
    else:
        raise Exception("Norm:", norm, "not defined")
    if gradient:
        delta.requires_grad = True
    return delta

def update(delta, grad, alpha, norm):
    if norm == "linf":
        d = delta + alpha * torch.sign(grad)
    elif norm == "l2":
        grad_l2norm = torch.norm(grad, p=2, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        scaled_grad = grad / (grad_l2norm + 1e-10)
        d = delta + alpha * scaled_grad
    else:
        raise Exception("Norm:", norm, "not defined")
    return d

def project(X, delta, eps, norm):

    if norm == "linf":
        d = torch.clamp(delta, -eps, eps)
    elif norm == "l2":
        grad_l2norm = torch.norm(delta, p=2, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        scaled_grad = delta / (grad_l2norm + 1e-10) * eps
        mask = (grad_l2norm > eps).squeeze()
        d = delta
        if mask.sum() > 0:
            d[mask] = scaled_grad[mask]
    else:
        raise Exception("Norm:", norm, "not defined")
    return d
