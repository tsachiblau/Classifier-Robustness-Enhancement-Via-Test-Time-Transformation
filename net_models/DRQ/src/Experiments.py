from src.models.Models import get_model
from .Test import test
from .Utils import get_attack_key

# Test all attacks on a model
def test_model(conf):
    model = get_model(conf)

    for attack in conf.attacks:
        key, counter_key = get_attack_key(attack["key"], conf, attack_conf=attack)
        if conf.test:
            test(conf, model, attack["type"], key, counter_key, attack)
