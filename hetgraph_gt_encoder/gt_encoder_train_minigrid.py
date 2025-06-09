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
os.makedirs("checkpoints_minigrid", exist_ok=True)


def get_data_for_state(st_loader, batch, all_aff_keys, all_obj_keys, action_keys, full_state):
    feats = batch[0][0]
    nei_index = batch[0][1]
    mps = st_loader.generate_mps_st(nei_index, all_aff_keys, all_obj_keys, action_keys, full_state)
    return feats, nei_index, mps

def get_data_for_episode(st_loader, batch, full_state, use_action_repr):
    feats = batch[0][0]
    nei_index = batch[0][1]
    mps = st_loader.generate_mps_episode(nei_index, full_state, use_action_repr)
    return feats, nei_index, mps


def run_epoch(st_loader, model, optimiser, logger, epoch, best, table, accumulation_steps, alpha,loss_type, nr_steps=200, use_action_repr=False):
    epoch_loss = 0
    # Initialize cumulative loss and set gradients to zero
    cumulative_loss = 0.0
    count_zeros = 0
    for i in range(nr_steps):
        (batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, (
        fstate_p1, fstate_p2, fstate_n1) = st_loader.get_subgraph_episode_data_minigrid(batch_size=1)
        feats_p1, nei_index_p1, mps_p1 = get_data_for_episode(st_loader, batch_pos1, fstate_p1, use_action_repr)
        feats_p2, nei_index_p2, mps_p2 = get_data_for_episode(st_loader, batch_pos2, fstate_p2, use_action_repr)
        feats_n1, nei_index_n1, mps_n1 = get_data_for_episode(st_loader, batch_neg1, fstate_n1, use_action_repr)
        if len(nei_index_n1) == 0 or len(nei_index_p1) == 0 or len(nei_index_p2) == 0:
            continue
        all_feats = (feats_p1, feats_p2, feats_n1)
        all_nei_index = (nei_index_p1, nei_index_p2, nei_index_n1)
        all_mps = (mps_p1, mps_p2, mps_n1)

        # optimiser.zero_grad()
        loss = model(all_feats, all_mps,all_nei_index, alpha, loss_type)
        logger.log({"Loss": loss.item()})
        cumulative_loss += loss.item()
        epoch_loss += loss.item()

        # logger.log({'pos_pair_t1': tp1, 'pos_pair_t2': tp2, 'neg_pair': tn})
        # table = wandb.Table(data=[[epoch, i, tp1, tp2, tn]], columns=['epoch','time_step', 'pos_pair_1', 'pos_pair_2', 'neg_pair'])
        # wandb.log({'timestep': f'Timestep_{i}', "time_postive": f'tp_1:{tp1}', "time_positive_2": f'tp_2:{tp2}', "time_neg": f'neg:{tn}'})
        # table.add_data(epoch, i, fstate_p1['Episode'][0][0],fstate_p2['Episode'][0][0],fstate_n1['Episode'][0][0])
        if loss.item() == 0:
            count_zeros += 1

        # logger.log({'times': table})

        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            # Update parameters
            optimiser.step()
            # Zero the gradients for the next accumulation_steps
            model.zero_grad()
            optimiser.zero_grad()
            # Print or log the cumulative loss
            print("cum loss ", cumulative_loss)
            logger.log({"Cumulative Loss": cumulative_loss})
            if cumulative_loss < best:
                best = cumulative_loss
                torch.save(model.state_dict(), f'checkpoints_minigrid2/ep_{loss_type}_{alpha}_{epoch}_{loss:.4f}.pkl')

            # Reset cumulative loss
            cumulative_loss = 0.0

    if nr_steps % accumulation_steps != 0:
        optimiser.step()
        model.zero_grad()
    logger.log({"Epoch Loss": epoch_loss/200})
    return count_zeros


def train(alpha, cummulation, loss_type, count_mps=2):
    tasks = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-UnlockPickup-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-SimpleCrossingS9N1-v0"
    ]
    st_loader = StateLoader(nr_mps=2, mps=None, use_memory='ltm2', tasks=tasks)
    (batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, (fstate_p1, fstate_p2, fstate_n1) = st_loader.get_subgraph_episode_data_minigrid(batch_size=1)
    feats = batch_pos1[0][0]
    nei_index = batch_pos1[0][1]
    mps = st_loader.generate_mps_episode(nei_index,  fstate_p1, has_action_repr=False)
    mps_dims = [mp.shape[1] for mp in mps]
    feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]

    wandb_logger = Logger(f"{alpha}_hybrid_loss_alpha_cummulation_20", project='episode_minigrid_encoder')
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
    table = wandb.Table(columns=["epoch", "time_step", 'ep_a','ep_p','ep_n'])
    model.train()
    all_zeros_run = 0
    for epoch in range(args.nb_epochs):
        zeros_epoch = run_epoch(st_loader, model, optimiser, logger, epoch, best,table, cummulation, alpha, loss_type)
        all_zeros_run = all_zeros_run + zeros_epoch
    logger.log({"Zeros Run": all_zeros_run})
    # logger.log({"Table": table})


if __name__ == '__main__':
    alphas = [ 1.5,1.2,1.0,0.5,0.2]
    cummulation = [10, 5]
    loss_type=['contrastive', 'triplet']
    for alpha in alphas:
        print(f"---------------alpha:{alpha}------------------")

        train(alpha=alpha, cummulation=20, loss_type='hybrid')

    # for i in alphas:
    #     for j in cummulation:
    #         for loss in loss_type:
    #             wandb_logger = Logger(f"{loss}_loss_alpha_{i}_cummulation_{j}", project='graph_encoder')
    #
    #             train(wandb_logger, alpha=i, cummulation=j, loss_type=loss)
