import os

os.environ['NEO4J_BOLT_URL'] = 'bolt://localhost:7687'
os.environ['NEO_PASS'] = 'rl123456'
os.environ['NEO_USER'] = 'neo4j'

import torch
from hetgraph_gt_encoder.data_helpers.data_preparation import StateLoader
from hetgraph_gt_encoder.models.HeCo import HeCo
from hetgraph_gt_encoder.heco_params import heco_params

count_mps = 2
args = heco_params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st_loader = StateLoader(nr_mps=2, mps=None)
(batch_pos1, batch_pos2, batch_neg1), all_state_keys, all_aff_keys, all_obj_keys, action_keys, (
fstate_p1, fstate_p2, fstate_n1) = st_loader.get_subgraph_episode_data(batch_size=1)
feats = batch_pos1[0][0]
nei_index = batch_pos1[0][1]
mps = st_loader.generate_mps_episode(nei_index, fstate_p1)
mps_dims = [mp.shape[1] for mp in mps]
feats_dim_list = [i.shape[1] for i in batch_pos1[0][0]]


def get_data_for_episode(st_loader, batch, full_state):
    feats = batch[0][0]
    nei_index = batch[0][1]
    mps = st_loader.generate_mps_episode(nei_index, full_state)
    return feats, nei_index, mps


def get_encoding_task_incp(model, task, succ):
    alpha = 0.5
    loss_type = None
    push_succ_1, all_state_keys, all_aff_keys, all_obj_keys, action_keys, fstate_push_succ_p1, anchor_file_push_succ_1 = st_loader.get_subgraph_episode_data_by_task_incp(
        task_1=task, succ_fail=succ)
    feats_push_succ_1, nei_index_push_succ_1, mps_push_succ_1 = get_data_for_episode(st_loader, push_succ_1,
                                                                                     fstate_push_succ_p1)
    z_sc_push_succ_1, z_mp_push_succ_1, intra_push_succ_1, inter_push_succ_1 = model(feats_push_succ_1, mps_push_succ_1,
                                                                                     nei_index_push_succ_1, alpha,
                                                                                     loss_type, testing=True)
    return z_sc_push_succ_1, anchor_file_push_succ_1


def get_encoding_task(model, task, succ):
    alpha = 0.5
    loss_type = None
    push_succ_1, all_state_keys, all_aff_keys, all_obj_keys, action_keys, fstate_push_succ_p1, anchor_file_push_succ_1 = st_loader.get_subgraph_episode_data_by_task(
        task_1=task, succ_fail=succ)
    feats_push_succ_1, nei_index_push_succ_1, mps_push_succ_1 = get_data_for_episode(st_loader, push_succ_1,
                                                                                     fstate_push_succ_p1)
    z_sc_push_succ_1, z_mp_push_succ_1, intra_push_succ_1, inter_push_succ_1 = model(feats_push_succ_1, mps_push_succ_1,
                                                                                     nei_index_push_succ_1, alpha,
                                                                                     loss_type, testing=True)
    return z_sc_push_succ_1, anchor_file_push_succ_1


dirs = ['checkpoints_12_cum20', 'checkpoints_15_cum20', 'checkpoints_hybrid']

checkpoints = []
for check_dir in dirs:
    PATH_TO_CHECKPOINTS = os.path.join(str(os.getcwd()), check_dir)
    files = os.listdir(PATH_TO_CHECKPOINTS)

    if check_dir == 'checkpoints_hybrid':
        file = files[33]
        print(files[33])
        print(file)
    else:
        print(files[21])
        file = files[21]

    checkpoint_path = os.path.join(PATH_TO_CHECKPOINTS, file)
    checkpoints.append(checkpoint_path)

print(checkpoints)


def compute_encodings_incp(checkpoint):
    tasks = ["CreateLevelPush-v0",
             "CreateLevelBuckets-v0",
             "CreateLevelBasket-v0", "CreateLevelBelt-v0",
             "CreateLevelObstacle-v0"]
    succ_fail = ['true', 'false']
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                 count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims).to(device)
    model.load_state_dict(torch.load(checkpoint))

    model.eval()
    z_encodings = []
    labels = []
    # Loop through tasks and outcomes to fetch episodes
    for task in tasks:
        for outcome in succ_fail:
            for _ in range(num_episodes_per_type):
                z_sc, anchor_file = get_encoding_task_incp(model, task, outcome)
                z_encodings.append(z_sc)
                labels.append((task, outcome))
                # try:
                #     z_sc, anchor_file = get_encoding_task_incp(model, task, outcome)
                #     z_encodings.append(z_sc)
                #     labels.append((task, outcome))
                # except:
                #     print(task)
                #     continue

    # Convert list of tensors to a tensor for efficient operations
    if len(z_encodings) > 0:
        z_encodings = torch.stack(z_encodings)
    return z_encodings, labels


def compute_encodings(checkpoint):
    tasks = ["CreateLevelPush-v0",
             "CreateLevelBuckets-v0",
             "CreateLevelBasket-v0", "CreateLevelBelt-v0",
             "CreateLevelObstacle-v0"]
    succ_fail = ['true', 'false']
    model = HeCo(args.hidden_dim, feats_dim_list, args.feat_drop, args.attn_drop,
                 count_mps, args.sample_rate, args.nei_num, args.tau, args.lam, mps_dims).to(device)
    model.load_state_dict(torch.load(checkpoint))

    model.eval()
    z_encodings = []
    labels = []
    # Loop through tasks and outcomes to fetch episodes
    for task in tasks:
        for outcome in succ_fail:
            for _ in range(num_episodes_per_type):
                try:
                    z_sc, anchor_file = get_encoding_task(model, task, outcome)
                    z_encodings.append(z_sc)
                    labels.append((task, outcome))
                except Exception as e:
                    print(e)
                    print(task)
                    continue

    # Convert list of tensors to a tensor for efficient operations
    z_encodings = torch.stack(z_encodings)
    return z_encodings, labels


import torch
import torch.nn.functional as F
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def compute_clusters_by_outcome(z_encodings, labels, tasks):
    # Convert tensor to numpy for sklearn compatibility
    z_encodings_np = z_encodings.cpu().detach().numpy()
    z_encodings_np = z_encodings_np.squeeze(1)  # Ensure the encodings are 2D: [num_samples, feature_dim]

    # Prepare dataframes for success and failure episodes
    df = pd.DataFrame({
        'task': [label[0] for label in labels],  # Original task names
        'outcome': [label[1] for label in labels]  # Success/Failure outcome
    })

    # Filter the encodings based on success and failure
    success_indices = df[df['outcome'] == 'true'].index
    failure_indices = df[df['outcome'] == 'false'].index

    z_success = z_encodings_np[success_indices]
    z_failure = z_encodings_np[failure_indices]

    # Define number of clusters as the number of unique tasks
    n_clusters = len(tasks)

    # Perform K-means clustering on successful episodes
    kmeans_success = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels_success = kmeans_success.fit_predict(z_success)

    # Create a DataFrame for successful episodes with their cluster labels
    df_clusters_success = pd.DataFrame({
        'task': df.loc[success_indices, 'task'].values,
        'cluster': cluster_labels_success
    })

    # Perform K-means clustering on unsuccessful episodes
    kmeans_failure = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels_failure = kmeans_failure.fit_predict(z_failure)

    # Create a DataFrame for unsuccessful episodes with their cluster labels
    df_clusters_failure = pd.DataFrame({
        'task': df.loc[failure_indices, 'task'].values,
        'cluster': cluster_labels_failure
    })

    # Visualize the clusters for successful episodes using PCA
    pca_success = PCA(n_components=2)
    z_success_pca = pca_success.fit_transform(z_success)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=z_success_pca[:, 0], y=z_success_pca[:, 1],
        hue=df_clusters_success['task'],
        palette='tab10', s=100
    )
    plt.title('K-means Clustering of Successful Episodes (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Task')
    plt.tight_layout()
    plt.show()

    # Visualize the clusters for unsuccessful episodes using PCA
    pca_failure = PCA(n_components=2)
    z_failure_pca = pca_failure.fit_transform(z_failure)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=z_failure_pca[:, 0], y=z_failure_pca[:, 1],
        hue=df_clusters_failure['task'],
        palette='tab10', s=100
    )
    plt.title('K-means Clustering of Unsuccessful Episodes (PCA Reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Task')
    plt.tight_layout()
    plt.show()

    # Evaluate clustering by task for success
    print("Cluster counts per task for successful episodes:")
    print(df_clusters_success.groupby(['task', 'cluster']).size())

    # Evaluate clustering by task for failure
    print("\nCluster counts per task for unsuccessful episodes:")
    print(df_clusters_failure.groupby(['task', 'cluster']).size())

    return df_clusters_success, df_clusters_failure


z_encodings_diff_checkpoints_incp = []
z_encoding_labels_incp = []
num_episodes_per_type = 5

z_enc, labels = compute_encodings_incp(checkpoints[2])
z_encoding_labels_incp.append(labels)
z_encodings_diff_checkpoints_incp.append(z_enc)
