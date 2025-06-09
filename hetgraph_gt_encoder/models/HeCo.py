import torch
import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, mp_dims, minigrid=False):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, mp_dims, attn_drop, minigrid=minigrid)
        self.sc = Sc_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def apply_model_to_set(self, feats, mps, nei_index):
        h_all = []
        for i in range(len(feats)):
            if len(feats[i]) > 0:
                h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_sc, all_intra_att, type_lvl_att = self.sc(h_all, nei_index)
        z_mp = self.mp(self.sc.h_refer, h_all, mps)
        return z_sc, z_mp, all_intra_att, type_lvl_att

    def forward(self, feats, mps, nei_index, alpha, loss_type, testing=False):  # p a s
        if testing:
            z_sc_1_p, z_mp_1_p, all_intra_att, type_lvl_att = self.apply_model_to_set(feats, mps, nei_index)
            return z_sc_1_p, z_mp_1_p, all_intra_att, type_lvl_att

        f1_p, f2_p, f3_n = feats
        mp1_p, mp2_p, mp3_n = mps
        nei1_p, nei2_p, nei3_n = nei_index
        z_sc_1_p, z_mp_1_p, _, _ = self.apply_model_to_set(f1_p, mp1_p, nei1_p)
        z_1 = torch.cat((z_sc_1_p, z_mp_1_p), dim=1)

        z_sc_2_p, z_mp_2_p,  _, _ = self.apply_model_to_set(f2_p, mp2_p, nei2_p)
        z_2 = torch.cat((z_sc_2_p, z_mp_2_p), dim=1)

        z_sc_1_n, z_mp_1_n,  _, _ = self.apply_model_to_set(f3_n, mp3_n, nei3_n)
        z_1n = torch.cat((z_sc_1_n, z_mp_1_n), dim=1)



        if loss_type == 'triplet':
            loss = self.contrast.triplet_loss(z_1, z_2, z_1n, alpha)

        elif loss_type == 'hybrid':
            loss = self.contrast.triplet_loss(z_1, z_2, z_1n, alpha)
            inter = self.contrast.cross_view_loss(z_sc_1_p, z_sc_2_p, z_mp_1_p, z_mp_2_p, z_mp_1_n)
            final_loss = 0.3*inter+0.7*loss
            loss = final_loss


        else:
            loss = self.contrast.contrastive_loss_combined(z_1, z_2, z_1n)


        # loss = self.contrast.contrastive_loss((z_sc_1_p, z_mp_1_p), (z_sc_2_p, z_mp_2_p), (z_sc_1_n, z_mp_1_n))
        return loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()
