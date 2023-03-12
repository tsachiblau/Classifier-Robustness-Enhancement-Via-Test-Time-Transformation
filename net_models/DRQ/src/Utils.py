import torch

def is_string(val):
    try:
        str(val)
        return True
    except:
        return False

def is_float(val):
    try:
        float(val)
        return True
    except:
        return False

def string_fraction_to_float(string):
    try:
        split = string.split("/")
        val = float(split[0]) / float(split[1])
        return val
    except:
        try:
            return float(string)
        except:
            return None

def get_attack_key(key, conf, attack_conf=None):
    if not (attack_conf is None):
        for dict_key in attack_conf:
            if dict_key != "key" and dict_key != "type" and dict_key != "name":
                key += "_" + str(attack_conf[dict_key])
    counter_key = ""
    if conf.counterattack_defense and len(conf.counterattack_defense) > 0:
        for dict_key in conf.counterattack_defense:
            if isinstance(conf.counterattack_defense[dict_key], dict):
                for dict_key_2 in conf.counterattack_defense[dict_key]:
                    counter_key += str(conf.counterattack_defense[dict_key][dict_key_2]) + ","
            elif dict_key != "name":
                counter_key += str(conf.counterattack_defense[dict_key]) + ","
        counter_key = "_cd(" + counter_key[:-1].replace("/", "-") + ")"
    return key.replace(":", ""), counter_key

def calc_acc(pred, y):
    acc = torch.sum(pred.argmax(1) == y)
    return acc.item()

def l2_norm(x):
    return torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1)

def format_string(s):
    return s.replace("_", " ").title()
