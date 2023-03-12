import copy

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import wandb
from datetime import datetime
import os
from functools import partial
import numpy as np
from attacks import get_threat_model, get_defense_threat_model
from get_dataset import get_data_loader
from get_num_of_classes import get_num_of_classes
from time import time

def eval_model(model, args):
    dict_KEYS = ['logits', 'MSE', 'MAE', 'LPIPS']
    os.makedirs('images', exist_ok=True)
    now = datetime.now()
    str_date_time = now.strftime("%d-%m-%Y_%H:%M")
    run_name = str(args.net_name) + '_arch_dataset_' + str(args.dataset) + '_threat_' + str(args.net_threat_model)

    #set wandb run
    wandb.init(project='reverse_pgd', config=args)
    wandb.run.name = run_name
    testloader = get_data_loader(args)
    attack_pgd = get_threat_model(args)

    ## we are doing eval of clean acc ##
    dict_eval = {}
    dict_eval['num_samples'] = 0
    dict_eval['num_vanila_correct'] = 0
    dict_eval['num_clear_defense_correct_distance_logits'] = 0
    dict_eval['num_clear_defense_correct_distance_MSE'] = 0
    dict_eval['num_clear_defense_correct_distance_MAE'] = 0
    dict_eval['num_clear_defense_correct_distance_LPIPS'] = 0
    dict_eval['num_default_correct'] = 0
    dict_eval['num_defense_correct_distance_MSE'] = 0
    dict_eval['num_defense_correct_distance_logits'] = 0
    dict_eval['num_defense_correct_distance_MAE'] = 0
    dict_eval['num_defense_correct_distance_LPIPS'] = 0

    N_quick_review = 400
    if args.dataset == 'imagenet':
        N_quick_review = 1000

    list_vanila_time = []
    list_defense_time = []
    list_top_k = []
    pbar = tqdm(testloader)
    for idx, (tuple_data) in enumerate(pbar):
        if len(tuple_data) == 2:
            x, y = tuple_data
            x_clear = copy.copy(x)
        else:
            x, y, x_clear = tuple_data
        x, y, x_clear = x.cuda(), y.cuda(), x_clear.cuda()

        '''
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure()
        plt.imshow(x[idx].cpu().permute(1, 2, 0). numpy())
        plt.title('class: {}'.format(y[idx]))
        plt.show()
        '''
        ## clac clear images
        before_vanila = time()
        with torch.no_grad():
            bool_correct_vanila = model(x_clear).argmax(dim=1).__eq__(y)
        after_vanila = time()

        ## clac attacked images
        x_attacked = attack_pgd.get_adv_x(model, x, y)
        with torch.no_grad():
            logits = model(x_attacked)
            bool_correct_default = logits.argmax(dim=1).__eq__(y)

            idx_sort = logits.sort(dim=1, descending=True)[1]
            tensor_bool_top_k = (idx_sort == y.unsqueeze(1)).int().argmax(dim=1)
            list_top_k += tensor_bool_top_k.tolist()

        # bool_correct_clear_dist_MSE = bool_correct_defense_dist_MSE = bool_correct_default
        ### reversed PGD on clear images
        dict_clear_dist_bool = {}
        dict_clear_dist = model(x_clear, use_defense=True)
        for key_i in dict_KEYS:
            dict_clear_dist_bool[key_i] = torch.zeros_like(bool_correct_vanila)
            if key_i in dict_clear_dist.keys():
                dict_clear_dist_bool[key_i] = dict_clear_dist[key_i].argmax(dim=1).__eq__(y)

        ### reverse PGD on attacked images
        dict_defense_dist_bool = {}
        before_defense = time()
        dict_defense_dist = model(x_attacked, use_defense=True)
        after_defense = time()

        for key_i in dict_KEYS:
            dict_defense_dist_bool[key_i] = torch.zeros_like(bool_correct_vanila)
            if key_i in dict_clear_dist.keys():
                dict_defense_dist_bool[key_i] = dict_defense_dist[key_i].argmax(dim=1).__eq__(y)

        # print(np.histogram(list_top_k, bins=get_num_of_classes(args), range=(0, get_num_of_classes(args)), density=True)[0])
        list_vanila_time.append(after_vanila-before_vanila)
        list_defense_time.append(after_defense-before_defense)
        print('vanila time: {},    defense time: {}'.format(np.mean(list_vanila_time), np.mean(list_defense_time)))
        ## eval the results
        for i in range(len(y)):
            dict_eval['num_samples'] += 1
            dict_eval['num_vanila_correct'] += bool_correct_vanila[i]
            dict_eval['num_default_correct'] += bool_correct_default[i]
            dict_eval['num_clear_defense_correct_distance_MSE'] += dict_clear_dist_bool['MSE'][i]
            dict_eval['num_defense_correct_distance_MSE'] += dict_defense_dist_bool['MSE'][i]
            dict_eval['num_clear_defense_correct_distance_logits'] += dict_clear_dist_bool['logits'][i]
            dict_eval['num_defense_correct_distance_logits'] += dict_defense_dist_bool['logits'][i]
            dict_eval['num_clear_defense_correct_distance_MAE'] += dict_clear_dist_bool['MAE'][i]
            dict_eval['num_defense_correct_distance_MAE'] += dict_defense_dist_bool['MAE'][i]
            dict_eval['num_clear_defense_correct_distance_LPIPS'] += dict_clear_dist_bool['LPIPS'][i]
            dict_eval['num_defense_correct_distance_LPIPS'] += dict_defense_dist_bool['LPIPS'][i]

            display_dict = {
                'vanila': dict_eval['num_vanila_correct'] / dict_eval['num_samples'],
                'clear defense dist logits': dict_eval['num_clear_defense_correct_distance_logits'] / dict_eval['num_samples'],
                'clear defense dist MSE': dict_eval['num_clear_defense_correct_distance_MSE'] / dict_eval['num_samples'],
                'clear defense dist MAE': dict_eval['num_clear_defense_correct_distance_MAE'] / dict_eval['num_samples'],
                'clear defense dist LPIPS': dict_eval['num_clear_defense_correct_distance_LPIPS'] / dict_eval['num_samples'],
                'defualt': dict_eval['num_default_correct'] / dict_eval['num_samples'],
                'defense cls dist logits': dict_eval['num_defense_correct_distance_logits'] / dict_eval['num_samples'],
                'defense cls dist MSE': dict_eval['num_defense_correct_distance_MSE'] / dict_eval['num_samples'],
                'defense cls dist MAE': dict_eval['num_defense_correct_distance_MAE'] / dict_eval['num_samples'],
                'defense cls dist LPIPS': dict_eval['num_defense_correct_distance_LPIPS'] / dict_eval['num_samples']
            }


            pbar.set_description(str(display_dict))
            wandb.log(display_dict)

            if args.quick_review and dict_eval['num_samples'] > N_quick_review:
                exit(0)

