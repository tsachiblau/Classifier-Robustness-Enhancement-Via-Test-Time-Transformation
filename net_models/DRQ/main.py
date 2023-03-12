#Both autoattack and FMN are from the github of the respective authors
#Small changes were made to FMN to give adversaries with specific confidence

from src.Configuration import Conf
from src.Experiments import test_model
import itertools
from read_yaml import open_yaml
import argparse

experiment_path = "your/experiment/path/"
data_path = "your/data/path"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_path", default=experiment_path)
    parser.add_argument("--data_path", default=data_path)
    parser.add_argument("--yaml_file", default=["cifar10c"], nargs='*')
    return parser.parse_args()

def main():
    input_args = get_args()
    for yaml_file in input_args.yaml_file:
        print("Yaml File:", yaml_file)
        arguments, keys = open_yaml(yaml_file)
        runs = list(itertools.product(*arguments))
        for r, run in enumerate(runs):
            current_args = {}
            current_args["experiment_path"] = input_args.experiment_path
            current_args["data_path"] = input_args.data_path
            for i in range(len(keys)):
                current_args[keys[i]] = run[i]
            conf = Conf(current_args)

            test_model(conf)

if __name__ == "__main__":
    main()
