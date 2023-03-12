# Classifier-Robustness-Enhancement-Via-Test-Time-Transformation

This repo contains the official implementation for the paper [Classifier-Robustness-Enhancement-Via-Test-Time-Transformation] (TODO) by Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad and Alex Bronstein. 


![alt text]([https://github.com/tsachiblau/Classifier-Robustness-Enhancement-Via-Test-Time-Transformation/edit/main/method.png](https://github.com/tsachiblau/Classifier-Robustness-Enhancement-Via-Test-Time-Transformation/blob/main/method.png)

## Preparting the code
In order to run the code you need to:

1. Install necassary packages
2. Download checkpoints and locate them under /models

### Necassary packages
lpips
matplotlib
AutoAttack
PyYAML
eagerpy
foolbox
wandb

### Checkpoints

For the other methods you should take the checkpoints from their repo
Madry et al.   [1]    Link: https://github.com/MadryLab/robustness  
Zhang et al.   [2]    Link: https://github.com/cassidylaidlaw/perceptual-advex  
Rebuffi et al. [3]    Link: https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness  
Gowal et al.   [4]    Link: https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness  



## Running Experiments
To evaluate different defenses you should:

1. Run auto attack on the base classfifier, and save it the attack localy 
2. Evaluate the model using different defenses

### Run AutoAttack
Run the following command, the attacked images is saved in under /data/.../*.pt

python main.py  --flow create_aa
                --net_name [at, rebuffi, gowal]
                --net_threat_model [Linf_8, L2_0.5]  
                --dataset [cifar10, cifar100, imagenet] 
                --attack_info [Linf_8, Linf_16, L2_0.5, L2_1.0] 
                --batch_size 128
                
### Evaluate
For evaluation, run the following command
python main.py  --flow create_aa
                --net_name [at, rebuffi, gowal]
                --net_threat_model [Linf_8, L2_0.5]  
                --dataset [cifar10, cifar100, imagenet] 
                --attack_info [Linf_8, Linf_16, L2_0.5, L2_1.0] 
                --batch_size 128
                --defense_method [TETRA, DRQ]
                --defense_gamma FLOAT
                --defense_alpha FLOAT
                
