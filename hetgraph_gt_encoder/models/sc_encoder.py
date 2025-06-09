import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        type_lvl_att = beta.data.cpu().numpy()
        # print("sc ", beta.data.cpu().numpy())  # type-level attention
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc, type_lvl_att


class intra_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()



    # def forward(self, nei, h, h_refer):
    #     #nei_emb = F.embedding(nei, h)
    #     if len(h.shape) == 1:
    #         h = h.unsqueeze(0)
    #     nei_emb = h.unsqueeze(0)
    #     h_refer = torch.unsqueeze(h_refer, 1)
    #     h_refer = h_refer.expand_as(nei_emb)
    #     all_emb = torch.cat([h_refer, nei_emb], dim=-1)
    #     attn_curr = self.attn_drop(self.att)
    #     att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
    #     att = self.softmax(att)
    #     # print(att)
    #     nei_emb = (att*nei_emb).sum(dim=1)
    #     return nei_emb, att

    def forward(self, nei, h, h_refer):
        # Embed the neighbors
        # nei_emb = F.embedding(nei, h)
        nei_emb = h.unsqueeze(0)

        # Use the learnable h_refer directly, expanding its dimensions to match nei_emb
        h_refer_expanded = h_refer.unsqueeze(0).expand_as(nei_emb)

        # Concatenate the reference embedding with neighbor embeddings
        all_emb = torch.cat([h_refer_expanded, nei_emb], dim=-1)

        # Apply attention mechanism
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)

        # Compute the weighted sum of neighbor embeddings
        nei_emb = (att * nei_emb).sum(dim=1)
        return nei_emb, att


class Sc_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop):
        super(Sc_encoder, self).__init__()
        self.intra = nn.ModuleList([intra_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = inter_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        self.h_refer = nn.Parameter(torch.empty(size=(hidden_dim,)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.h_refer.unsqueeze(0))


    def forward(self, nei_h, nei_index):
        embeds = []
        all_intra_att = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            # for per_node_nei in nei_index[i]:
            #     if len(per_node_nei) >= sample_num:
            #         select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
            #                                                    replace=False))[np.newaxis]
            #     else:
            #         select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,
            #                                                    replace=True))[np.newaxis]
            #     sele_nei.append(select_one)
            # sele_nei = torch.cat(sele_nei, dim=0).cuda()
            intra_att_emb, att = self.intra[i](nei_index[i], nei_h[i], self.h_refer)
            one_type_emb = F.elu(intra_att_emb)
            all_intra_att.append(att)
            embeds.append(one_type_emb)
        z_mc, type_lvl_att = self.inter(embeds)
        return z_mc, all_intra_att, type_lvl_att
