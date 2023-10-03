import argparse
import os
import torch
import time

from affordance_learning.affordance_data import AffDataset
from affordance_learning.neural_statistician import Statistician
from memory_graph.memory_utils import WorkingMemory, ConceptSpaceGDS

from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

from baseline_models.logger import Logger


# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Aff Experiment')

# required
parser.add_argument('--data-dir', type=str, default='create_aff_ds',
                    help='location of formatted Omniglot data')
parser.add_argument('--output-dir', type=str, default='checkpoints_ns_aff',
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size (of datasets) for training (default: 64)')
parser.add_argument('--sample-size', type=int, default=5,
                    help='number of sample images per dataset (default: 5)')
parser.add_argument('--c-dim', type=int, default=512,
                    help='dimension of c variables (default: 512)')
parser.add_argument('--n-hidden-statistic', type=int, default=1,
                    help='number of hidden layers in statistic network modules '
                         '(default: 1)')
parser.add_argument('--hidden-dim-statistic', type=int, default=1000,
                    help='dimension of hidden layers in statistic network (default: 1000)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=16,
                    help='dimension of z variables (default: 16)')
parser.add_argument('--n-hidden', type=int, default=1,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 1)')
parser.add_argument('--hidden-dim', type=int, default=1000,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 1000)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for training (default: 300)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=2,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
args = parser.parse_args()


def get_aff_emb_context_and_instance(model, optimizer, object_type, datasets):
    cwd = os.getcwd()
    path = os.path.join(cwd, 'checkpoints_ns_aff/checkpoints/ns_78.ckp')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.eval()

    data = datasets.get_object_by_type(object_type).unsqueeze(0)

    context_latent, obj_instance, recon_img = model(data)
    return context_latent, obj_instance, recon_img


def compute_dist_within_ds(v):
    pdist = torch.nn.PairwiseDistance(p=2)
    all_dist = []
    for i in v:
        for j in v:
            dist = pdist(i, j)
            all_dist.append(dist.item())
    return all_dist

def compute_dist_diff_ds(v1, v2):
    pdist = torch.nn.PairwiseDistance(p=2)
    return pdist(v1, v2)


def add_test_data():
    # create datasets
    wm = WorkingMemory(which_db="afftest")
    concept_space = ConceptSpaceGDS(memory_type="afftest")

    train_dataset = AffDataset(data_dir=args.data_dir, split='train',
                                            n_frames_per_set=5)
    datasets = (train_dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    loaders = (train_loader)

    # create model
    n_features = 256 * 4 * 4  # output shape of convolutional encoder
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.elu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    variation_nr = 20
    ds_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "create_aff_ds\\train")
    object_types = os.listdir(ds_path)
    contexts = {}
    for otype in object_types:
        #contexts[otype] = {'list': [], 'dist': None}
        for i in range(variation_nr):
            context, inst, img = get_aff_emb_context_and_instance(model, optimizer, otype, datasets)
            aff_context = concept_space.add_data('ActionRepr')
            aff_id = aff_context['elementId(n)'][0]
            concept_space.set_property(aff_id, 'ActionRepr', 'val', context.squeeze(0).tolist())
            concept_space.set_property(aff_id, 'ActionRepr', 'obj_type', f'"{otype}"')



    return contexts

def view_stats_of_data():
    wm = WorkingMemory(which_db="afftest")
    name = wm.create_query_graph('afftest', 'ActionRepr', ['val'])
    clusters = wm.compute_action_clusters(f'"{name}"')
    return clusters



if __name__ == '__main__':
    view_stats_of_data()
