import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import random
import numpy as np
from sklearn.model_selection import train_test_split


from tqdm import tqdm

from models import RNNGenerator, RNNDiscriminator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN:
    def __init__(self,
                 latent_code_size,
                 hidden_size,
                 generator_output_size,
                 discriminator_output_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 max_len):
        print("[INFO] Init GAN model")
        self.latent_code_size = latent_code_size
        self.hidden_size = hidden_size
        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.max_len = max_len

        self.generator = RNNGenerator(
            latent_code_size, hidden_size, generator_output_size, num_layers, dropout).to(device)
        self.discriminator = RNNDiscriminator(
            generator_output_size, hidden_size, discriminator_output_size, num_layers, bidirectional, dropout).to(device)

    def fit(self,
            dataset,
            epochs,
            batch_size,
            learning_rate,
            patience,
            k_step,
            test_size,
            pretrain_epochs,
            teacher_forcing_rate):
        print("[INFO] Start fitting model")
        # train_set, valid_set = train_test_split(
        #     dataset, test_size=test_size, random_state=1)
        # convert to tensor
        train_set = dataset
        # valid_set = list(map(torch.FloatTensor, valid_set))
        # get dataloader
        train_loader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=1)
        # valid_loader = DataLoader(
        #     valid_set, batch_size, shuffle=True, num_workers=1)

        #######################
        ### TRAINING CONFIG ###
        #######################
        self.hist = {'pretrain': {'loss': []},
                     'train': {'gen_loss': [], 'dis_loss': []}}
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=learning_rate)
        self.pretrain_criterion = nn.MSELoss()
        #################
        ### PRE-TRAIN ###
        #################
        print("[INFO] Pre-Train model {} epochs on {} training samples".format(
            pretrain_epochs, len(train_set)))
        for epoch in range(pretrain_epochs):
            sum_loss = 0
            for target_batch in tqdm(train_loader):
                use_teacher_forcing = True if random.random() < teacher_forcing_rate else False
                target_batch = target_batch.to(device)
                latent_code_batch = self.sample_latent_code_batch(
                    batch_size, self.max_len, self.latent_code_size).to(device)
                init_hidden = self.generator.zero_hidden_state(
                    batch_size).to(device)
                sum_loss += self.pretrain_step(
                    target_batch, latent_code_batch, init_hidden, use_teacher_forcing)
            batch_loss = np.divide(sum_loss, len(train_loader))
            self.hist['pretrain']['loss'].append(batch_loss)
            print(
                "[INFO] Pre-Train Epoch {} -- MSE Loss: {:.6f}".format(epoch+1, batch_loss))
        print("[INFO] Pre-Train finished")
        #########################
        ### Adversarial Train ###
        #########################
        for epoch in range(epochs):
            ###########################
            ### Train Discriminaotr ###
            ###########################
            for k in range(k_step):
                latent_code_batch = self.sample_latent_code_batch(batch_size, self.max_len, self.latent_code_size).to(device)
                negative_labels = torch.zeros(batch_size, self.discriminator_output_size)
                positive_examples_batch = self.sample_examples_batch(train_set, batch_size)
                positive_labels = torch.ones(batch_size, self.discriminator_output_size)
                
    
    def sample_examples_batch(self, dataset, batch_size):
        sampled_idxs = np.random.choice(len(dataset), batch_size)
        sampled_examples = torch.cat([dataset[i].unsqueeze(0) for i in sampled_idxs])
        return sampled_examples


    def pretrain_step(self,
                      target_batch,
                      latent_code_batch,
                      init_hidden,
                      use_teacher_forcing):
        """Pre-train using concatenatation of teacher forcing input and latent code input with MSE loss"""
        self.generator.train()
        self.generator_optimizer.zero_grad()
        target_batch.transpose_(0, 1)
        latent_code_batch.transpose_(0, 1)
        first_motion_input = target_batch[0]
        # minus 1 because the output of last time step is useless
        if use_teacher_forcing:
            all_motion_outputs = self.generate_sequence(
                first_motion_input, latent_code_batch, init_hidden, teacher_sequence=target_batch)
        else:
            all_motion_outputs = self.generate_sequence(
                first_motion_input, latent_code_batch, init_hidden, teacher_sequence=None)
        loss = self.pretrain_criterion(target_batch, all_motion_outputs)
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def generate_sequence(self,
                          first_motion_input,
                          latent_code_batch,
                          init_hidden,
                          teacher_sequence=None):
        all_motion_outputs = torch.zeros(0, latent_code_batch.size(
            1), self.generator_output_size).to(device)  # variable to stroe all outputs
        motion_input = first_motion_input
        all_motion_outputs = torch.cat(
            [all_motion_outputs, motion_input.unsqueeze(0)])
        latent_input = latent_code_batch[0]
        hidden = init_hidden
        if teacher_sequence is not None:
            for step in range(1, self.max_len):
                output, hidden = self.generator.generate_step(
                    motion_input, latent_input, hidden)
                all_motion_outputs = torch.cat(
                    [all_motion_outputs, output.unsqueeze(0)])
                motion_input = teacher_sequence[step]
                latent_input = latent_code_batch[step]
            return all_motion_outputs
        else:
            for step in range(1, self.max_len):
                output, hidden = self.generator.generate_step(
                    motion_input, latent_input, hidden)
                all_motion_outputs = torch.cat(
                    [all_motion_outputs, output.unsqueeze(0)])
                motion_input = output
                latent_input = latent_code_batch[step]
            return all_motion_outputs

    def sample_latent_code_batch(self, time_step, batch_size, size):
        """Normal distribution random value"""
        return torch.rand(time_step, batch_size, size)
