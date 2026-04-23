import os
import torch
import time
import numpy as np
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from tools.statistics import GWADStatistics, AttackStatistics
from tools.attack_model import AttackModel as ATTACK_MODEL
from gwad import GWAD
from algorithm.attack.base import Query


def make_grid(sample):
    img = torchvision.utils.make_grid(sample)
    img = img.cpu()
    return img.detach().numpy()


class Statistics:
    def __init__(self):
        self.gwad = GWADStatistics()
        self.attack = AttackStatistics()

    def increment_runs(self):
        self.gwad.increment_runs()
        self.attack.increment_runs()


def show_predicrions(gwad_def, hx_record, total_queries):
    hx = gwad_def.get_predictions()
    hx = hx.numpy()
    queries, delta_queries = gwad_def.get_queries()
    total_queries += delta_queries

    #print("GWAD_defence : delta-net predicted {} HoDS during {} queries".format(
    #    int(np.sum(hx)), queries
    #))
    #print("GWAD_defence : accumulated predictions {}/{}".format(
       # int(np.sum(hx_record) + np.sum(hx)), total_queries
    #))
    print("GWAD_defence : delta-net predicted {} HoDS during {} queries".format(
    int(np.sum(hx)), queries
    ))

    print("GWAD_defence : accumulated delta-net calls = {}".format(
    total_queries
    ))
    for i in range(len(hx_record)):
        hx_record[i] += hx[i]
        print("%s[%d], " % (gwad_def.classes[i], hx_record[i]), end='')
    print("")

    return total_queries


def save_distributions(stats_dir, model, stats, feat_type, data_type):
    os.makedirs(f"{stats_dir}/hist", exist_ok=True)
    os.makedirs(f"{stats_dir}/dist", exist_ok=True)

    hist = stats.gwad.mean_hist()
    hist_file = f"{stats_dir}/hist/{data_type}_{model.name}_{feat_type}"
    np.savetxt(f"{hist_file}.txt", hist, delimiter=',')

    dist = stats.gwad.distribution
    dist_file = f"{stats_dir}/dist/{data_type}_{feat_type}_{stats.gwad.runs}"
    np.savetxt(f"{dist_file}.txt", dist, delimiter=',')


def benign(device, cfg, data_type, model, delta_net, loader):
    stats_dir = 'stats'
    cfg_gwad = cfg["gwad"]
    attack_name = 'benign'

    stats = Statistics()

    gwad_def = GWAD(
        device, cfg_gwad, stats.gwad,
        mode='defence',
        model=model,
        delta_net=delta_net
    )

    gwad_def.reset()

    hx_record = np.zeros(len(delta_net.classes))
    total_queries = 0

    for idx, (data, true_class) in enumerate(loader):
        if idx >= 100:
            break

        data = data.to(device)
        q = Query(t='benign', x=data)
        gwad_def.run(q)

    stats.increment_runs()

    show_predicrions(gwad_def, hx_record, total_queries)
    stats.gwad.show_screen()
    save_distributions(stats_dir, model, stats, attack_name, data_type)


def attack(device, cfg, data_type, model, delta_net, loader , train_loader):
    stats_dir = 'stats'
    cfg_attack, cfg_gwad = cfg["attack"], cfg["gwad"]

    attack_name = cfg_attack["name"]
    adapt_type = cfg_attack["adaptive"]["name"]
    batch_size = cfg_attack["adaptive"]["batch_size"]
    pool_size = cfg_attack["adaptive"]["pool_size"]
    move_rate = cfg_attack["adaptive"]["move_rate"]
    q_budgets = [cfg_attack["query_budget"]]

    stats = Statistics()

    gwad_def = GWAD(
        device, cfg_gwad, stats.gwad,
        mode='defence',
        model=model,
        delta_net=delta_net
    )

    gwad_evl = GWAD(
        device, cfg_gwad, stats.gwad,
        mode='evaluate',
        model=model,
        delta_net=delta_net
    )

    print("\nsequence of queries : attack \n{} {} {} {}".format(
        data_type, model.name, attack_name, delta_net.name
    ))

    hx_record = np.zeros(len(delta_net.classes))
    total_gwad_predict = 0
    total_attack_queries = 0
    x2_pool = []
    for pool_data, _ in train_loader:
        idx = torch.randperm(pool_data.size(0))
        for i in idx:
            x2_pool.append(pool_data[i].unsqueeze(0).to(device))
            if len(x2_pool) >= pool_size:
               break
            if len(x2_pool) >= pool_size:
               break
 
    cnt = 0

    for idx, (data, true_class) in enumerate(loader):
        if idx >= 10000:
            break

        data, true_class = data.to(device), true_class.to(device)

        if adapt_type == 'batch':
            if len(x2_pool) < pool_size:
                x2_pool.append(data)
                continue

        gwad_def.reset()

        attack = ATTACK_MODEL(
            data_type,
            device,
            stats.attack,
            cfg_attack,
            d_model=gwad_def.run,
            q_budgets=q_budgets,
            stop=False
        )

        attack.set_adaptive(adapt_type, move_rate, batch_size, x2_pool)

        t0 = time.perf_counter()
        adv = attack.run(data, true_class)
        t1 = time.perf_counter()

        adv_img = adv[0]
        if adv_img.dim() == 3:
            adv_img = adv_img.unsqueeze(0)

        q = Query(t='attack', x=adv_img)
        predict = gwad_evl.run(q)

        attack.update_stats(data, adv_img, true_class, predict)

        stats.increment_runs()

        total_gwad_predict = show_predicrions(
            gwad_def, hx_record, total_gwad_predict
        )

        print("Attack stats : num - time [true] : [adv][val/suc, dist, ratio, [i1] [i2] [i3]]")
        print("{} - {:.3f}s [{}] : ".format(
            stats.gwad.runs, t1 - t0, true_class.item()
        ), end='')

        stats.attack.show_stats()

        cnt += 1
        total_attack_queries += stats.attack.iter0[0]

        print("average attack queries - {}".format(
            int(total_attack_queries / cnt)
        ))

        stats.gwad.show_screen()

        save_distributions(stats_dir, model, stats, attack_name, data_type)
