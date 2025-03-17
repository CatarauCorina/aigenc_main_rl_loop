import numpy
import torch
import wandb
from hetgraph_gt_encoder.data_helpers.data_preparation import StateLoader
from hetgraph_gt_encoder.models.HeCo import HeCo
from hetgraph_gt_encoder.heco_params import heco_params
import warnings
import datetime
import pickle as pkl
import os
import random
from baseline_models.logger import Logger

warnings.filterwarnings('ignore')
args = heco_params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_data_for_state(st_loader, batch, all_aff_keys, all_obj_keys, action_keys, full_state):
    feats = batch[0][0]
    nei_index = batch[0][1]
    mps = st_loader.generate_mps_st(nei_index, all_aff_keys, all_obj_keys, action_keys, full_state)
    return feats, nei_index, mps


def run_epoch(st_loader, model, optimiser, logger, epoch, best, table, accumulation_steps, alpha,loss_type, nr_steps=200):
    epoch_loss = 0
    # Initialize cumulative loss and set gradients to zero
    cumulative_loss = 0.0
    for i in range(nr_steps):
        (batch_pos1, batch_pos2, batch_neg1), key, all_aff_keys, all_obj_keys, action_keys, (
        fstate_p1, fstate_p2, fstate_n1), (tp1, tp2, tn) = st_loader.get_subgraph_state_data(batch_size=1)
        feats_p1, nei_index_p1, mps_p1 = get_data_for_state(st_loader, batch_pos1, all_aff_keys, all_obj_keys,
                                                            action_keys, fstate_p1)
        feats_p2, nei_index_p2, mps_p2 = get_data_for_state(st_loader, batch_pos2, all_aff_keys, all_obj_keys,
                                                            action_keys, fstate_p2)
        feats_n1, nei_index_n1, mps_n1 = get_data_for_state(st_loader, batch_neg1, all_aff_keys, all_obj_keys,
                                                            action_keys, fstate_n1)
        all_feats = (feats_p1, feats_p2, feats_n1)
        all_nei_index = (nei_index_p1, nei_index_p2, nei_index_n1)
        all_mps = (mps_p1, mps_p2, mps_n1)
        model.train()
        optimiser.zero_grad()
        loss = model(all_feats, all_nei_index, all_mps, alpha, loss_type)
        logger.log({"Loss": loss.item()})
        cumulative_loss += loss.item()
        epoch_loss += loss.item()

        # logger.log({'pos_pair_t1': tp1, 'pos_pair_t2': tp2, 'neg_pair': tn})
        # table = wandb.Table(data=[[epoch, i, tp1, tp2, tn]], columns=['epoch','time_step', 'pos_pair_1', 'pos_pair_2', 'neg_pair'])
        # wandb.log({'timestep': f'Timestep_{i}', "time_postive": f'tp_1:{tp1}', "time_positive_2": f'tp_2:{tp2}', "time_neg": f'neg:{tn}'})
        table.add_data(epoch, i, tp1, tp2, tn)

        # logger.log({'times': table})

        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            # Update parameters
            optimiser.step()
            # Zero the gradients for the next accumulation_steps
            model.zero_grad()
            # Print or log the cumulative loss
            print("loss ", loss.data.cpu())
            logger.log({"Cumulative Loss": cumulative_loss})
            if cumulative_loss < best:
                best = loss
                torch.save(model.state_dict(), f'{loss_type}_{alpha}_{epoch}_{loss}.pkl')

            # Reset cumulative loss
            cumulative_loss = 0.0

    if nr_steps % accumulation_steps != 0:
        optimiser.step()
        model.zero_grad()
    logger.log({"Epoch Loss": epoch_loss/200})
    return


def train(wandb_logger, alpha, cummulation, loss_type, count_mps=2):
    st_loader = StateLoader(nr_mps=2, mps=None)
    (batch_pos1, batch_pos2, batch_neg1), key, all_aff_keys, all_obj_keys, action_keys, (fstate_p1, fstate_p2, fstate_n1), t = st_loader.get_subgraph_state_data_no_sim_check(batch_size=1)
    feats = batch_pos1[0][0]
    nei_index = batch_pos1[0][1]
    mps = st_loader.generate_mps_st(nei_index, all_aff_keys, all_obj_keys, action_keys, fstate_p1)
    mps_dims = [mp.shape[1] for mp in mps]
    feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]

    logger = wandb_logger.get_logger()
    print("seed ", args.seed)
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", mps)

    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                 count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims).to(device)
    model.zero_grad()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    starttime = datetime.datetime.now()
    best = 1e9
    table = wandb.Table(columns=["epoch", "time_step","pos1","pos2","neg1"])

    for epoch in range(args.nb_epochs):
        run_epoch(st_loader, model, optimiser, logger, epoch, best,table, cummulation, alpha, loss_type)
    logger.log({"Table": table})


if __name__ == '__main__':
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5]
    cummulation = [10, 5]
    loss_type=['contrastive', 'triplet']
    wandb_logger = Logger(f"zoomed_triplet_loss_alpha_cummulation", project='graph_encoder')
    train(wandb_logger, alpha=0.5, cummulation=5, loss_type='triplet')

    # for i in alphas:
    #     for j in cummulation:
    #         for loss in loss_type:
    #             wandb_logger = Logger(f"{loss}_loss_alpha_{i}_cummulation_{j}", project='graph_encoder')
    #
    #             train(wandb_logger, alpha=i, cummulation=j, loss_type=loss)
