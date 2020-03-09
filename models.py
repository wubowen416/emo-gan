import torch
import torch.nn as nn


class RNNGenerator(nn.Module):
    def __init__(self,
                 latent_code_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 dropout):
        print("[INFO] Init RNNGenerator")
        super(RNNGenerator, self).__init__()
        self.latent_code_size = latent_code_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # init layers
        self.pre_fc = nn.Linear(output_size + latent_code_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size,
                          num_layers=num_layers, dropout=dropout)
        self.post_fc = nn.Linear(hidden_size, output_size)

    def generate_step(self, motion_input, latent_code, hidden=None):
        """Generate each time step based on motion inputs and hidden state
        Args:
            motion_inputs -- in shape (B, O)
            latent_code -- in shape (B, L)
            hidden -- in shape (num_layers*num_direction, B, H)
        """
        pre_input = torch.cat(
            [motion_input, latent_code], dim=-1)  # (B, H+L)
        rnn_input = torch.relu(self.pre_fc(
            pre_input).unsqueeze(0))  # (T=1, B, H+L)
        # (T=1, B, H), (nl*nd, B, H)
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        output = self.post_fc(rnn_output).squeeze(0)  # (B, O)
        return output, hidden

    def zero_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class RNNDiscriminator(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 bidirectional,
                 dropout):
        print("[info] Init RNNDiscriminator")
        super(RNNDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        # init layers
        self.pre_fc = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.post_fc = nn.Linear(hidden_size, output_size)

    def discriminate_step(self, motion_input, hidden=None):
        """Discriminate one step of a sequence
        Args:
            motion_inputs -- in shape (B, I)
            hidden -- in shape (num_layers*num_direction, B, H)
        """
        rnn_input = torch.relu(self.pre_fc(motion_input)
                               ).unsqueeze(0)  # (1, B, H)
        rnn_output, hidden = self.rnn(
            rnn_input, hidden)  # (1, B, H), (nl*nd, B, H)
        output = self.post_fc(rnn_output).squeeze(0)  # (B, O)
        return output, hidden

    def zero_hidden_state(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
