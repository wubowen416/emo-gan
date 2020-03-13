from models import RNNGenerator, RNNDiscriminator
from custom_loss import DLoss, GLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN:
    def __init__(self,
                 latent_code_size,
                 hidden_size,
                 generator_output_size,
                 discriminator_output_size,
                 num_layers,
                 bidirectional,
                 relu_slope,
                 dropout,
                 max_len):
        print("[INFO] Init GAN model")
        self.latent_code_size = latent_code_size
        self.hidden_size = hidden_size
        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.relu_slope = relu_slope
        self.max_len = max_len

        self.generator = RNNGenerator(
            latent_code_size, hidden_size, generator_output_size, num_layers, self.relu_slope, dropout).to(device)
        self.discriminator = RNNDiscriminator(
            generator_output_size, hidden_size, discriminator_output_size, num_layers, bidirectional, self.relu_slope, dropout).to(device)

    def fit(self,
            dataset,
            epochs,
            batch_size,
            sample_size, 
            learning_rate,
            k_step,
            pretrain_epochs,
            teacher_forcing_rate,
            save_path):
        print("[INFO] Start fitting model")
        self.batch_size = batch_size
        self.sample_size = sample_size
        # convert to tensor
        train_set = dataset
        train_loader = DataLoader(
            train_set, batch_size, shuffle=True, num_workers=1)
        #######################
        ### TRAINING CONFIG ###
        #######################
        self.hist = {'pretrain': {'loss': []},
                     'train': {'d0_loss': [], 'd0_acc': [], 'd1_loss': [], 'd1_acc': [], 'g_loss': []}}
        self.pretrain_optimizer = optim.Adam(
            self.generator.parameters(), lr=learning_rate['pretrain'])
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=learning_rate['g'])
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate['d'])
        self.pretrain_criterion = nn.MSELoss()
        self.discriminator_criterion = DLoss()
        self.generator_criterion = GLoss()
        #################
        ### PRE-TRAIN ###
        #################
        if pretrain_epochs:
            print("[INFO] Pre-Train model {} epochs on {} training samples".format(
                pretrain_epochs, len(train_set)))
            for epoch in range(pretrain_epochs):
                sum_loss = 0
                for target_batch in tqdm(train_loader):
                    use_teacher_forcing = True if random.random() < teacher_forcing_rate else False
                    target_batch = target_batch.to(device)
                    latent_code_batch = self._sample_latent_code_batch(
                        batch_size, self.max_len, self.latent_code_size).to(device)
                    init_hidden = self.generator.zero_hidden_state(
                        batch_size).to(device)
                    sum_loss += self._pretrain_step(
                        target_batch, latent_code_batch, init_hidden, use_teacher_forcing)
                batch_loss = np.divide(sum_loss, len(train_loader))
                print(
                    "[INFO] Pre-Train Epoch {} -- MSE Loss: {:.6f}".format(epoch+1, batch_loss))
                # save log
                self.hist['pretrain']['loss'].append(batch_loss)
                torch.save(self.generator.state_dict(),
                           save_path + '_generator_pretrain.pt')
            print("[INFO] Pre-Train finished")
        # save generated result of pre-train
        _, generated_examples_batch = self._sample_pos_neg_examples(
            train_set, batch_size)
        result = generated_examples_batch.transpose(
            0, 1).cpu().detach().numpy()
        np.save('/home/wu/projects/emo-gan/generated/pretrain_result.npy', result)
        #########################
        ### Adversarial Train ###
        #########################
        print("[INFO] Start Adversarial Training")
        add_label = True
        for epoch in range(epochs):
            ###########################
            ### Train Discriminaotr ###
            ###########################
            d0_sum_loss = 0
            d0_sum_acc = 0
            d1_sum_loss = 0
            d1_sum_acc = 0
            for k in range(k_step):
                positive_examples_batch, negative_examples_batch = self._sample_pos_neg_examples(
                    train_set, batch_size)
                loss, acc = self._discriminator_train_step(
                    positive_examples_batch, negative_examples_batch)
                d0_sum_loss += loss['n_loss']
                d0_sum_acc += acc['n_acc']
                d1_sum_loss += loss['p_loss']
                d1_sum_acc += acc['p_acc']
            d0_loss = np.divide(d0_sum_loss, k_step)
            d0_acc = np.divide(d0_sum_acc, k_step)
            d1_loss = np.divide(d1_sum_loss, k_step)
            d1_acc = np.divide(d1_sum_acc, k_step)
            #######################
            ### Train Generator ###
            #######################
            _, generated_examples_batch = self._sample_pos_neg_examples(
                train_set, batch_size)
            # train step
            g_loss = self._generator_train_step(
                generated_examples_batch)
            # generate examples using trained generator
            _, generated_examples_batch = self._sample_pos_neg_examples(
                train_set, batch_size)
            print("[INFO] Epoch {} -- d0_loss: {:.6f}, d0_acc: {:.2f}, d1_loss: {:.6f}, d1_acc: {:.2f}, g_loss: {:.6f}".format(
                epoch+1, d0_loss, d0_acc, d1_loss, d1_acc, g_loss))
            # save log
            self.hist['train']['d0_loss'].append(d0_loss)
            self.hist['train']['d0_acc'].append(d0_acc)
            self.hist['train']['d1_loss'].append(d1_loss)
            self.hist['train']['d1_acc'].append(d1_acc)
            self.hist['train']['g_loss'].append(g_loss)
            df = pd.DataFrame(self.hist['train'])
            df.to_csv('/home/wu/projects/emo-gan/chkpt/log/log.csv')
            torch.save(self.generator.state_dict(),
                       save_path + '_generator.pt')
            torch.save(self.discriminator.state_dict(),
                       save_path + '_discriminator.pt')
            # plot learning curve
            fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150)
            ax1.set_title('loss')
            ax1.plot(self.hist['train']['d0_loss'], label='d-fake')
            ax1.plot(self.hist['train']['d1_loss'], label='d-real')
            ax1.plot(self.hist['train']['g_loss'], label='gen')
            ax1.grid()
            ax1.legend()
            ax2.set_title('acc')
            ax2.plot(self.hist['train']['d0_acc'], label='fake')
            ax2.plot(self.hist['train']['d1_acc'], label='real')
            ax2.grid()
            ax2.legend()
            add_label = False
            plt.savefig('curve.png')
            plt.close()
            # save generated sequence every 100 epochs
            if epoch % 1000 == 0:
                # save generated result of adversarial train
                result = generated_examples_batch.transpose(
                    0, 1).cpu().detach().numpy()
                np.save(
                    '/home/wu/projects/emo-gan/generated/adversarial_result_epoch_{}.npy'.format(epoch), result)

    def discriminate_sequence(self, seq_batch, init_hidden):
        all_outputs = torch.zeros(
            0, seq_batch.size(1), self.discriminator_output_size).to(device)
        hidden = init_hidden
        for seq_step in seq_batch:
            output, hidden = self.discriminator.discriminate_step(
                seq_step, hidden)
            all_outputs = torch.cat([all_outputs, output.unsqueeze(0)])
        mean_output = torch.mean(all_outputs, dim=0)
        return mean_output, hidden

    def generate_sequence(self,
                          first_motion_input,
                          latent_code_batch,
                          init_hidden,
                          teacher_sequence=None):
        """Generate a complete sequence using generator
        Args:
            first_motion_input -- shape (T=1, B, O)
            latent_code_batch -- shape (T=max_len, B, L)
            init_hidden -- shape (nl*nd, B, H)
            teacher_sequence -- A torch tensor, (T, B, O)
            if exists, generate using teacher forcing input,
            if not, infer with self output
        """
        all_motion_outputs = torch.zeros(0, first_motion_input.size(0), self.generator_output_size).to(
            device)  # variable to stroe all outputs
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
            return all_motion_outputs, hidden
        else:
            for step in range(1, self.max_len):
                output, hidden = self.generator.generate_step(
                    motion_input, latent_input, hidden)
                all_motion_outputs = torch.cat(
                    [all_motion_outputs, output.unsqueeze(0)])
                motion_input = output
                latent_input = latent_code_batch[step]
            return all_motion_outputs, hidden

    def _generator_train_step(self, generated_examples_batch):
        self.generator_optimizer.zero_grad()
        init_hidden = self.discriminator.zero_hidden_state(
            self.batch_size).to(device)
        output_batch, _ = self.discriminate_sequence(
            generated_examples_batch, init_hidden)
        loss = self.generator_criterion(output_batch)
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def _discriminator_train_step(self, positive_examples_batch, negative_examples_batch):
        self.discriminator_optimizer.zero_grad()
        init_hidden = self.discriminator.zero_hidden_state(
            self.batch_size).to(device)
        positive_output_batch, _ = self.discriminate_sequence(
            positive_examples_batch, init_hidden)
        negative_output_batch, _ = self.discriminate_sequence(
            negative_examples_batch, init_hidden)
        loss, p_loss, n_loss = self.discriminator_criterion(
            positive_output_batch, negative_output_batch)
        loss.backward()
        self.discriminator_optimizer.step()
        # create labels for samples and
        # calculate acc
        positive_labels_batch = torch.ones(
            self.batch_size,  dtype=torch.long).to(device)
        negative_labels_batch = torch.zeros(
            self.batch_size,  dtype=torch.long).to(device)
        # positive
        positive_output_batch = positive_output_batch.view(-1)
        positive_predicted_batch = (positive_output_batch > 0.5).float()
        positive_correct_count = (
            positive_predicted_batch == positive_labels_batch).sum().item()
        positive_acc = np.divide(positive_correct_count, self.batch_size)
        # negative
        negative_output_batch = negative_output_batch.view(-1)
        negative_predicted_batch = (negative_output_batch > 0.5).float()
        negative_correct_count = (
            negative_predicted_batch == negative_labels_batch).sum().item()
        negative_acc = np.divide(negative_correct_count, self.batch_size)
        return {'p_loss': p_loss.item(), 'n_loss': n_loss.item()}, \
            {'p_acc': positive_acc, 'n_acc': negative_acc}

    def _pretrain_step(self,
                       target_batch,
                       latent_code_batch,
                       init_hidden,
                       use_teacher_forcing):
        """Pre-train,
        using concatenatation of teacher forcing input and latent code input,
        with MSE loss
        """
        self.generator.train()
        self.generator_optimizer.zero_grad()
        target_batch.transpose_(0, 1)
        latent_code_batch.transpose_(0, 1)
        first_motion_input = target_batch[0]
        # minus 1 because the output of last time step is useless
        if use_teacher_forcing:
            all_motion_outputs, _ = self.generate_sequence(
                first_motion_input, latent_code_batch, init_hidden, teacher_sequence=target_batch)
        else:
            all_motion_outputs, _ = self.generate_sequence(
                first_motion_input, latent_code_batch, init_hidden, teacher_sequence=None)
        loss = self.pretrain_criterion(all_motion_outputs, target_batch)
        loss.backward()
        self.generator_optimizer.step()
        return loss.item()

    def _sample_pos_neg_examples(self, dataset, batch_size):
        real_examples_batch = self._sample_examples_batch(
            dataset, batch_size).transpose(0, 1).to(device)
        first_motion_input = real_examples_batch[0]
        latent_code_batch = self._sample_latent_code_batch(
            batch_size, self.max_len, self.latent_code_size).transpose(0, 1).to(device)
        init_hidden = self.generator.zero_hidden_state(
            batch_size).to(device)
        generated_examples_batch, _ = self.generate_sequence(
            first_motion_input, latent_code_batch, init_hidden, teacher_sequence=None)
        return real_examples_batch, generated_examples_batch

    def _sample_examples_batch(self, dataset, batch_size):
        """Randomly choose number of batch_size samples from dataset"""
        sampled_idxs = np.random.choice(len(dataset), batch_size)
        sampled_examples = torch.cat(
            [dataset[i].unsqueeze(0) for i in sampled_idxs])
        return sampled_examples

    def _sample_latent_code_batch(self, batch_size, time_step, latent_code_size):
        """Normal distribution random value"""
        return torch.rand(batch_size, time_step, latent_code_size)
