from src.DecisionRegionQuantification import decision_region_robustness
from src.Datasets import get_dataset_information
from src.CustomEnums import DataSetName

import robustbench as rb
import torch.nn as nn

def get_model(conf):

    dir = conf.model_save_path("")
    model_name = conf.model.name.replace("_L2", "")

    if conf.dataset == DataSetName.cifar10 or conf.dataset == DataSetName.cifar10c:
        dir = dir.replace("cifar10/" + conf.model_norm + "/" + conf.model.name.replace("_L2", "") + ".pt", "")
        model = rb.utils.load_model(model_name=model_name, dataset="cifar10", model_dir=dir, norm=conf.model_norm)
    else:
        raise Exception("Only CIFAR10 is implemented for now")
    model = model.cuda()
    model = counterattack_model(conf, model)

    return model

class counterattack_model(nn.Module):
    def __init__(self, conf, model):
        super(counterattack_model, self).__init__()
        self.model = model
        self.counterattack_parameters = conf.counterattack_defense
        self.conf = conf
        self.do_counter_attack = False
        self.classes = get_dataset_information(conf.dataset)["classes"]

    def forward(self, X):
        return self.model(X)

    def forward_decision_region(self, X, counterattack_parameters=None):
        return decision_region_robustness(self, X, self.classes, counterattack_parameters)

