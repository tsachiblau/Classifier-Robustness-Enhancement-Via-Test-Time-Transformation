from net_models.DRQ.src.CustomEnums import *

train_keys = ['model', 'dataset',
              'batch_size', 'epochs',
             'seed', "pretrained"]
test_keys = ["test", "experiment_path", "data_path",
             'model_norm', 'attacks', "denoise", "preprocess_denoise",
            'counterattack_defense', 'corruptions', 'severity', 'validate', 'ood_dataset',
             'table_name', 'viable_conf_keys', 'table_row', 'table_col', 'corner_value', 'worst_case',
             'highlight_results', 'keep_only_worst_case', 'add_mean_row', 'plots', "test_calibration"]
optional = ["seed"]
optional_test = ["result_path"]

all_enums = {ModelName, DataSetName, AdversarialAttacks, LossFunction, Norm}

enums = {"model", "dataset", "loss_function", "norm_in", "norm_out", "attacks", 'ood_dataset', 'plots'}

def get_run(config):
    def run(config2):

        print(config, config2)


# run1 = get_run("bla")

from torch import nn

# class Pipeline:
#     def __init__(self, nn.Module, loss, Scaling):
#         x=1
#
#     def run(self, data):
#         self.scaling.train(data)
#
#         # Data preprocess
#         self.results = self.algo.run(data)
#
#         return self