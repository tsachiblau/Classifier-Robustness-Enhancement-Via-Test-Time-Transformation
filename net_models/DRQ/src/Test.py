import torch
import torch.optim

from .Adversarial import get_adversarial_attack
from .Evaluation import save_result_dict
from .Datasets import get_data_set
from .Utils import calc_acc

def test(conf, model, attack_type, key, counter_key, attack_conf):
    total_key = key + counter_key
    print("Testing: {}".format(total_key))

    test_loader = get_data_set(conf)
    counterattack_parameters = conf.counterattack_defense
    model.eval()

    n = 0
    acc_sum = 0
    acc_drq_sum = 0

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        delta = get_adversarial_attack(conf, model, attack_type, X, y, attack_conf)
        X_adv = (X + delta).detach()

        if counter_key == "":
            with torch.no_grad():
                pred = model(X_adv)
                acc_org = calc_acc(pred, y) * 100
        else:
            pred, delta_counter, distances, original_pred, epsilon = model.forward_decision_region(X_adv, counterattack_parameters)
            acc_drq = calc_acc(pred, y) * 100
            acc_org = calc_acc(original_pred, y) * 100
            acc_drq_sum += acc_drq
        n += y.size(0)
        acc_sum += acc_org
        res_txt = f"Batch {i} acc: {acc_sum / n}"
        if counter_key != "":
            res_txt += f" acc_drq: {acc_drq_sum / n}"
        print(res_txt)

    result_dict = {}
    result_dict[total_key + " acc"] = acc_sum / n
    result_dict[total_key + " acc drq"] = acc_drq_sum / n
    save_result_dict(conf, result_dict, name=conf.model.name + "_metrics")