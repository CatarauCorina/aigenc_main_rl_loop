import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def contrastive_loss(self, z_i, z_j, z_l, temperature=0.3):
        # Split the embeddings into two parts
        zi1, zi2 = torch.split(z_i, [z_i.size(1) // 2, z_i.size(1) // 2], dim=1)
        zj1, zj2 = torch.split(z_j, [z_j.size(1) // 2, z_j.size(1) // 2], dim=1)
        zl1, zl2 = torch.split(z_l, [z_l.size(1) // 2, z_l.size(1) // 2], dim=1)

        # Calculate similarity scores for both parts
        sim_ij1 = F.cosine_similarity(zi1, zj1, dim=-1) / temperature
        sim_ij2 = F.cosine_similarity(zi2, zj2, dim=-1) / temperature
        sim_il1 = F.cosine_similarity(zi1, zl1, dim=-1) / temperature
        sim_il2 = F.cosine_similarity(zi2, zl2, dim=-1) / temperature

        # Combine similarity scores
        sim_ij = (sim_ij1 + sim_ij2) / 2
        sim_il = (sim_il1 + sim_il2) / 2

        # Calculate probabilities
        exp_sim_ij = torch.exp(sim_ij)
        exp_sim_il = torch.exp(sim_il)

        # Calculate the negative log probabilities
        neg_log_prob_ij = -torch.log(exp_sim_ij / (exp_sim_ij + exp_sim_il.sum(dim=-1)))
        neg_log_prob_il = -torch.log(exp_sim_il / (exp_sim_ij + exp_sim_il.sum(dim=-1)))

        # Calculate the overall contrastive loss
        contrastive_loss = neg_log_prob_ij + neg_log_prob_il

        return contrastive_loss.mean()

    def contrastive_loss_combined(self, z_i, z_j, z_l, temperature=0.3):

        # Calculate similarity scores for both parts
        pos_pair = self.sim(z_i, z_j) / temperature
        neg_pair = self.sim(z_i, z_l) / temperature

        # Calculate probabilities
        exp_sim_ij = torch.exp(pos_pair)
        exp_sim_il = torch.exp(neg_pair)

        # Calculate the negative log probabilities
        neg_log_prob_ij = -torch.log(exp_sim_ij / (exp_sim_ij + exp_sim_il.sum(dim=-1)))
        neg_log_prob_il = -torch.log(exp_sim_il / (exp_sim_ij + exp_sim_il.sum(dim=-1)))

        # Calculate the overall contrastive loss
        contrastive_loss = neg_log_prob_ij + neg_log_prob_il

        return contrastive_loss.mean()


    def cross_view_loss(self, z_sc_1_p, z_sc_2_p, z_mp_1_p, z_mp_2_p, z_mp_1_n, temperature=0.3):
        temperature = 0.5

        # Similarity between positive pairs (within and across views)
        pos_sim_1 = F.cosine_similarity(z_sc_1_p, z_mp_1_p) / temperature  # schema 1 and metapath 1 (same node)
        pos_sim_2 = F.cosine_similarity(z_sc_1_p, z_mp_2_p) / temperature  # schema 1 and metapath 2 (positive pair)
        pos_sim_3 = F.cosine_similarity(z_sc_2_p, z_mp_2_p) / temperature  # schema 2 and metapath 2 (same node)

        # Similarity between negative pairs
        neg_sim_1 = F.cosine_similarity(z_sc_1_p, z_mp_1_n) / temperature  # schema 1 and metapath negative
        neg_sim_2 = F.cosine_similarity(z_sc_2_p, z_mp_1_n) / temperature  # schema 2 and metapath negative

        # Positive similarities
        pos_loss = -torch.log(torch.exp(pos_sim_1) / (torch.exp(pos_sim_1) + torch.exp(neg_sim_1)))
        pos_loss += -torch.log(torch.exp(pos_sim_2) / (torch.exp(pos_sim_2) + torch.exp(neg_sim_2)))
        pos_loss += -torch.log(torch.exp(pos_sim_3) / (torch.exp(pos_sim_3) + torch.exp(neg_sim_2)))

        pos_loss = pos_loss.mean()
        return pos_loss

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def triplet_loss(self, anchor, positive, negative, alpha):
        triplet_l = nn.TripletMarginLoss(margin=alpha, p=2, eps=1e-7)
        return triplet_l(anchor, positive, negative)

    # def forward(self, z_mp, z_sc, pos):
    #     z_proj_mp = self.proj(z_mp)
    #     z_proj_sc = self.proj(z_sc)
    #     matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
    #     matrix_sc2mp = matrix_mp2sc.t()
    #
    #     matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
    #     lori_mp = -torch.log(matrix_mp2sc.mul(pos).sum(dim=-1)).mean()
    #
    #     matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
    #     lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
    #     return self.lam * lori_mp + (1 - self.lam) * lori_sc

    # def forward(self, z_mp, z_sc):
    #     z_proj_mp = self.proj(z_mp)
    #     z_proj_sc = self.proj(z_sc)
    #     matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
    #     matrix_sc2mp = matrix_mp2sc.t()
    #
    #     # Normalize the similarity matrices
    #     matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
    #     lori_mp = -torch.log(matrix_mp2sc.sum(dim=-1)).mean()
    #
    #     matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
    #     lori_sc = -torch.log(matrix_sc2mp.sum(dim=-1)).mean()
    #
    #     return self.lam * lori_mp + (1 - self.lam) * lori_sc

    import torch
    import torch.nn.functional as F

    def forward(self, z_mp, z_sc, labels, temperature=0.5):
        # Project embeddings
        z_proj_mp = self.proj(z_mp)  # Metapath view projection
        z_proj_sc = self.proj(z_sc)  # Schema view projection

        # Calculate similarity matrix between the two views
        similarity_matrix = torch.mm(z_proj_mp, z_proj_sc.t()) / temperature

        # Mask to keep only positive pairs based on labels
        # `labels` should be a matrix where `1` indicates positive pairs, `0` for negatives
        exp_sim = torch.exp(similarity_matrix)  # Exponentiate similarities
        pos_sim = exp_sim * labels  # Select positive similarities
        all_sim = exp_sim.sum(dim=-1, keepdim=True)  # Sum over all similarities

        # Calculate the positive similarity score as a fraction of all similarities
        pos_loss = -torch.log((pos_sim.sum(dim=-1) + 1e-8) / (all_sim + 1e-8)).mean()

        return pos_loss

