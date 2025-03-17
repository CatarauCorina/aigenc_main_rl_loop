import argparse
import os
import torch
import torchvision
import time

from affordance_learning.affordance_data import AffDataset
from affordance_learning.neural_statistician import Statistician
from memory_graph.memory_utils import WorkingMemory, ConceptSpaceGDS
from conv_lstm_triplet import ConvLSTM, EmbeddingNet

from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

from baseline_models.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class ActionEmbedder:

    def __init__(self, concept_space=None):
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
        cwd = os.getcwd()
        path = os.path.join(cwd, '..', 'affordance_learning/checkpoints_ns_aff/checkpoints/ns_78.ckp')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model.eval()
        self.model = model
        self.optimizer = optimizer
        if concept_space is not None:
            self.concept_space = concept_space

    def get_action_embedding(self, data):
        context_latent, obj_instance, recon_img = self.model(data)
        return context_latent, obj_instance, recon_img

    def add_action_to_concept_space(self, context, otype):
        aff_context = self.concept_space.add_data('ActionRepr')
        aff_id = aff_context['elementId(n)'][0]
        self.concept_space.set_property(aff_id, 'ActionRepr', 'val', context.squeeze(0).tolist())
        self.concept_space.set_property(aff_id, 'ActionRepr', 'obj_type', f'"{otype}"')
        return aff_id



def get_aff_emb_context_and_instance(model, optimizer, object_type, datasets):
    cwd = os.getcwd()
    path = os.path.join(cwd, 'checkpoints_ns_aff/checkpoints/embed_net_10batch_margin_3lay_49.ckp')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    model.eval()
    data, obj_id_type = datasets.get_object_by_type(object_type)

    data= data.unsqueeze(0)

    embedding = model(data)
    return embedding, obj_id_type


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



def check_if_node_exists(embedding, obj_type, gds, similarity_th=0.5, similarity_method='euclidean', node_type='ActionRepr'):
        similar_objects = gds.run_cypher(
            f"""
                match(no:{node_type}) with 
                    no, gds.similarity.{similarity_method}(
                        {embedding},
                        no.val
                    ) as sim
                where sim > {similarity_th} and no.obj_type = "{obj_type}"
                return elementId(no),no.obj_type, no.obj_id_type, no.obj_types, sim order by sim desc limit 6
            """
        )
        return similar_objects


def check_if_node_exists_update(embedding, gds, similarity_th=0.5):
    act_reprs = get_all_act_repr(gds)
    if len(act_reprs) == 0:
        return None
    val = torch.Tensor(list(act_reprs['val'])).to(device)

    normalized_tensor = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
    val = F.normalize(val, p=2, dim=1)

    # Step 2: Compute pairwise Euclidean distances
    distances = torch.cdist(normalized_tensor, val).squeeze(0)
    print(distances)

    distances_normalized = distances / distances.max()
    pairs = torch.nonzero(distances_normalized > similarity_th, as_tuple=True)
    distances_below_threshold = distances_normalized[pairs]
    if len(pairs) > 0 and  val.shape[0] == 1:
        return act_reprs['elementId(n)'][0]
    result = list(zip(pairs[0].tolist(), pairs[1].tolist(), distances_below_threshold.tolist()))
    if len(result) > 0:
        index = result[0][1]
        id_same_act = act_reprs['elementId(n)'][index]
        return id_same_act
    return None

def check_if_node_exists_cosine(tensor1, tensor2):
    tensor1 = tensor1.flatten()
    tensor2 = tensor2.flatten()
    return F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

def check_node_exists(embedding, gds, similarity_th=0.5):
    act_reprs = get_all_act_repr(gds)
    if len(act_reprs) == 0:
        return None
    val = torch.Tensor(list(act_reprs['val'])).to(device)
    similarities = [check_if_node_exists_cosine(embedding, t) for t in val]
    sim_highest = similarities.index(max(similarities))
    if max(similarities) > similarity_th:
        id_same_act = act_reprs['elementId(n)'][sim_highest]
        return id_same_act



def get_act_repr(gds):
        centroids = gds.run_cypher(
            """
                MATCH (n:ActionRepr) where n.obj_type='Ramp'
                return n.val as val, n.context as context, n.obj_type, n.obj_id_type
            """
        )
        return centroids


def get_all_act_repr(gds):
    centroids = gds.run_cypher(
        """
            MATCH (n:ActionRepr) 
            return elementId(n), n.val as val
        """
    )
    return centroids




def add_test_data():
    # create datasets
    wm = WorkingMemory(which_db="afftestnew")
    concept_space = ConceptSpaceGDS(memory_type="afftestnew")

    train_dataset = AffDataset(data_dir=args.data_dir, split='train',
                                            n_frames_per_set=5)
    datasets = (train_dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    loaders = (train_loader)

    # create model
    embedding_net = EmbeddingNet(input_dim=3, hidden_dim=[64, 128], kernel_size=(3, 3), num_layers=2, embedding_dim=16)
    embedding_net.to(device)
    optimizer = optim.Adam(embedding_net.parameters(), lr=0.0001)
    variation_nr = 5

    ds_path_new = os.path.join(os.path.dirname(os.getcwd()), "create_aff_ds\\objects_obs_grouped")

    object_types = os.listdir(ds_path_new)
    contexts = {}
    for otype in object_types:
        #contexts[otype] = {'list': [], 'dist': None}
        for i in range(variation_nr):
            embedding, obj_id_type = get_aff_emb_context_and_instance(embedding_net, optimizer, otype, datasets)
            act_repr_exists = check_if_node_exists( embedding.tolist()[0], otype, wm.gds, 0.55)

            if len(act_repr_exists) > 0:
                aff_id = act_repr_exists['elementId(no)'][0]
                # types = concept_space.get_property('ActionRepr', aff_id, 'obj_types')
                updated_types = act_repr_exists['no.obj_types'][0] + [otype]
                concept_space.set_property(aff_id, 'ActionRepr', 'obj_types', updated_types)
            else:
                aff_context = concept_space.add_data('ActionRepr')
                aff_id = aff_context['elementId(n)'][0]
                concept_space.set_property(aff_id, 'ActionRepr', 'val', embedding.tolist()[0])
                concept_space.set_property(aff_id, 'ActionRepr', 'obj_type', f'"{otype}"')
                concept_space.set_property(aff_id, 'ActionRepr', 'obj_id_type', f'"{obj_id_type}"')
                concept_space.set_property(aff_id, 'ActionRepr', 'obj_types', [])

                # else:




            # recog_img = img.cpu().squeeze(0)
            # for img in recog_img:
            #     img_plot = torchvision.transforms.functional.to_pil_image(img)
            #     import matplotlib.pyplot as plt
            #     plt.imshow(img_plot)
            #     plt.show()
            # aff_context = concept_space.add_data('ActionRepr')
            # aff_id = aff_context['elementId(n)'][0]
            # concept_space.set_property(aff_id, 'ActionRepr', 'val', inst[3].tolist())
            # concept_space.set_property(aff_id, 'ActionRepr', 'obj_type', f'"{otype}"')



    return contexts

def view_stats_of_data():
    add_test_data()
    # wm = WorkingMemory(which_db="afftestnew")
    # ramp_repr = get_act_repr(wm.gds)
    # val_tens = torch.Tensor(list(ramp_repr['val']))
    # context_tens = torch.Tensor(list(ramp_repr['context']))
    #
    # check_if_node_exists_update(torch.Tensor(list(ramp_repr['val'])[0]), wm.gds)


    # check_if_node_exists(ramp_repr['val'][0],wm.gds)
    # name = wm.create_query_graph('afftestnew2', 'ActionRepr', ['val, context'])
    # clusters = wm.compute_action_clusters(f'"{name}"')
    # return clusters



if __name__ == '__main__':
    view_stats_of_data()
