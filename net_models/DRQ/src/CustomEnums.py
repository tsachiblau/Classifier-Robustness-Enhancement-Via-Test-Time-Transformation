from enum import Enum

class ModelName(Enum):
    Wong2020Fast = 1
    Sehwag2020Hydra = 2
    Wang2020Improving = 3
    Hendrycks2019Using = 4
    Rice2020Overfitting = 5
    Zhang2019Theoretically = 6
    Engstrom2019Robustness = 7
    Chen2020Adversarial = 8
    Gowal2020Uncovering_28_10_extra = 9
    Huang2020Self = 10
    Pang2020Boosting = 11
    Carmon2019Unlabeled = 12
    Ding2020MMA = 13
    Zhang2019You = 14
    Zhang2020Attacks = 15
    Wu2020Adversarial_extra = 16
    Wu2020Adversarial = 17
    Augustin2020Adversarial_L2 = 18
    Engstrom2019Robustness_L2 = 19
    Rice2020Overfitting_L2 = 20
    Rony2019Decoupling_L2 = 21
    Ding2020MMA_L2 = 22
    Wu2020Adversarial_L2 = 23
    Mustafa2019 = 25
    JinRinard2020 = 26
    Gowal2020Uncovering_34_20 = 28
    Standard = 30
    Hendrycks2020AugMix_ResNeXt = 31
    Hendrycks2020AugMix_WRN = 32
    ImageNetRobustLibrary = 33
    ImageNetFastIsBetter = 34
    ImageNetDeepAugmentAndAugmix = 35

    def __str__(self):
        return self.name

class DataSetName(Enum):
    cifar10 = 1
    cifar10c = 2
    cifar100 = 3
    cifar100c = 4
    svhn = 5
    ImageNet_A = 6
    ImageNet_C = 7
    gaussian_noise = 8
    uniform_noise = 9
    imagenet = 10

    def __str__(self):
        return self.name

class AdversarialAttacks(Enum):
    normal = 0
    pgd = 1
    fgsm = 3
    noise_uniform = 4
    noise_gauss = 5
    apgd = 8
    rotation = 9
    fab = 10
    square = 11
    translation = 12

    def __str__(self):
        return self.name


class LossFunction(Enum):
    # Cross entropy
    ce = 0
    ce_scaled = 6

    def __str__(self):
        return self.name

class Norm(Enum):
    l2 = 0
    linf = 1

class ResultType(Enum):
    classification = 0
    attack = 1
    decision_region_robustness = 2
    start_epsilon = 3