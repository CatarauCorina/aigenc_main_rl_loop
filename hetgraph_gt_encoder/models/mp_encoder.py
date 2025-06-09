import torch
import torch.nn as nn


class ModifiedGCN(nn.Module):
    def __init__(self, in_state_ft, in_action_ft, out_ft, bias=True):
        super(ModifiedGCN, self).__init__()
        self.fc_state = nn.Linear(in_state_ft, out_ft, bias=False)
        self.fc_action = nn.Linear(in_action_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, state_features, action_features, state_action_adjacency):
        state_fts = state_features
        action_fts = self.fc_state(action_features)

        # Select only the connected actions
        connected_action_fts = action_fts

        # Aggregate connected action features
        out = state_fts + torch.sum(connected_action_fts, dim=0)

        if self.bias is not None:
            out += self.bias

        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
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
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp


class Mp_encoder(nn.Module):
    def __init__(self, P, hidden_dim, mp_dims, attn_drop, minigrid=False):
        super(Mp_encoder, self).__init__()
        self.P = P
        self.node_level = nn.ModuleList([ModifiedGCN(hidden_dim, mp_dims[i], hidden_dim) for i in range(P)])
        self.att = Attention(hidden_dim, attn_drop)
        self.minigrid= minigrid

    def forward(self, h_refer, h, mps):
        embeds = []
        if self.minigrid:
            num_paths = min(self.P, len(mps), len(h) - 1)
            for i in range(num_paths):
                embed = self.node_level[i](h_refer, h[i + 1], mps[i])
                embed = embed.unsqueeze(0)
                embeds.append(embed)
            z_mp = self.att(embeds)
            return z_mp

        for i in range(self.P):
            embed = self.node_level[i](h_refer, h[i+1], mps[i])
            embed = embed.unsqueeze(0)
            embeds.append(embed)
        z_mp = self.att(embeds)
        return z_mp
