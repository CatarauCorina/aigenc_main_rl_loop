import torch
import torch.nn as nn
import torch.optim as optim
from affordance_learning.affordance_data_triplet import AffDatasetTriplet
import argparse
from torch.utils import data
from baseline_models.logger import Logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (time, batch, channel, height, width) -> (batch, time, channel, height, width)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            pass
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0),
                                             image_size=(input_tensor.size(3), input_tensor.size(4)))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim[-1] * 64 * 64, embedding_dim)  # Adjust the size based on the final output dimensions

    def forward(self, x):
        outputs, _ = self.convlstm(x)
        final_output = outputs[0][:, -1, :, :, :]  # Take the last time step's output
        final_output = final_output.view(final_output.size(0), -1)  # Flatten the output
        embedding = self.fc(final_output)
        return embedding

# Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        negative_distance = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        loss = torch.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()


import os
def main():
    wandb_logger = Logger(f"conv_lstm_triplet_margin_0.5", project='action_reprs_conv_lstm')
    logger = wandb_logger.get_logger()
    embedding_net = EmbeddingNet(input_dim=3, hidden_dim=[64, 128], kernel_size=(3, 3), num_layers=2, embedding_dim=16)
    embedding_net.to(device)
    triplet_loss = TripletLoss(margin=0.6)
    optimizer = optim.Adam(embedding_net.parameters(), lr=0.0001)

    parser = argparse.ArgumentParser(description='Neural Statistician Aff Experiment')

    # required
    parser.add_argument('--data-dir', type=str, default='create_aff_ds',
                        help='location of formatted Omniglot data')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size (of datasets) for training (default: 64)')
    args = parser.parse_args()
    train_dataset = AffDatasetTriplet(data_dir=args.data_dir, split='train',
                                      n_frames_per_set=5)
    datasets = (train_dataset)

    # create loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        step = 0
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            anchor_embedding = embedding_net(anchor)
            positive_embedding = embedding_net(positive)
            negative_embedding = embedding_net(negative)
            loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            loss.backward()

            epoch_loss = epoch_loss + loss.item()
            print(f'Step {step} Loss  {loss.item()}')
            optimizer.step()
            step += 1
        epoch_loss = epoch_loss / len(train_dataset)
        logger.log({'Loss epoch': epoch_loss})

        PATH = f"embed_net_10batch_margin_3lay_{epoch+1}.ckp"
        save_path = os.path.join('checkpoints_ns_aff/','checkpoints/', PATH)
        torch.save({
            'model_state': embedding_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, save_path)
    return

if __name__ == '__main__':
    main()
