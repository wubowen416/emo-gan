from models import RNNGenerator, RNNDiscriminator
from custom_loss import DLoss, GLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import time
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
            sample_rate,
            pretrain_epochs,
            teacher_forcing_rate,
            save_path):
        print("[INFO] Start fitting model")
        self.batch_size = batch_size
        self.sample_size = sample_size
        n_step = int(np.ceil(sample_size / batch_size))
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
        self.pretrain_criterion = nn.MSELoss()
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(), lr=learning_rate['g'])
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate['d'])
        self.discriminator_criterion = DLoss()
        self.generator_criterion = GLoss()
        #################
        ### PRE-TRAIN ###
        #################
        if pretrain_epochs:
            print("[INFO] Pre-Train model {} epochs on {} training samples".format(
                pretrain_epochs, len(train_set)))
            
            for epoch in range(pretrain_epochs):
                start_time = time.time()
                sum_loss = 0
                for target_batch in train_loader:
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
                    "[INFO] Pre-Train Epoch {} - {:.2f}s: MSE Loss: {:.6f}".format(epoch+1, time.time()-start_time, batch_loss))
                # save log
                self.hist['pretrain']['loss'].append(batch_loss)
                torch.save(self.generator.state_dict(),
                           save_path + '_generator_pretrain.pt')
            print("[INFO] Pre-Train finished")
        # save generated result of pre-train
        _, generated_examples_batch = self._sample_pos_neg_examples(
            train_set, 10)  # 10 is how many samples to generated and save
        result = generated_examples_batch.transpose(
            0, 1).cpu().detach().numpy()
        rescaled_result = np.array(list(map(dataset.rescale, result)))
        np.save('/home/wu/projects/emo-gan/generated/pretrain_result.npy', result)
        #########################
        ### Adversarial Train ###
        #########################
        print("[INFO] Start Adversarial Training")
        add_label = True
        for epoch in range(epochs):
            start_time = time.time()
            ###########################
            ### Train Discriminaotr ###
            ###########################
            d0_sum_loss_n = 0
            d0_sum_acc_n = 0
            d1_sum_loss_n = 0
            d1_sum_acc_n = 0
            for _ in range(n_step):  # train model with n times batch-size number of examples
                d0_sum_loss_k = 0
                d0_sum_acc_k = 0
                d1_sum_loss_k = 0
                d1_sum_acc_k = 0
                for _ in range(k_step):
                    positive_examples_batch, negative_examples_batch = self._sample_pos_neg_examples(
                        train_set, batch_size)
                    loss, acc = self._discriminator_train_step(
                        positive_examples_batch, negative_examples_batch)
                    d0_sum_loss_k += loss['n_loss']
                    d0_sum_acc_k += acc['n_acc']
                    d1_sum_loss_k += loss['p_loss']
                    d1_sum_acc_k += acc['p_acc']
                d0_sum_loss_n += np.divide(d0_sum_loss_k, k_step)
                d0_sum_acc_n += np.divide(d0_sum_acc_k, k_step)
                d1_sum_loss_n += np.divide(d1_sum_loss_k, k_step)
                d1_sum_acc_n += np.divide(d1_sum_acc_k, k_step)
            d0_loss = np.divide(d0_sum_loss_n, n_step)
            d0_acc = np.divide(d0_sum_acc_n, n_step)
            d1_loss = np.divide(d1_sum_loss_n, n_step)
            d1_acc = np.divide(d1_sum_acc_n, n_step)
            #######################
            ### Train Generator ###
            #######################
            g_sum_loss_n = 0
            for _ in range(n_step):
                _, generated_examples_batch = self._sample_pos_neg_examples(
                    train_set, batch_size)
                # train step
                g_sum_loss_n += self._generator_train_step(
                    generated_examples_batch)
            g_loss = np.divide(g_sum_loss_n, n_step)
            # generate examples using trained generator and save
            _, generated_examples_batch = self._sample_pos_neg_examples(
                train_set, 10)  # 10 is how many samples to generated and save
            print("[INFO] Epoch {} - {:.1f}s: d0_loss: {:.6f}, d0_acc: {:.2f}, d1_loss: {:.6f}, d1_acc: {:.2f}, g_loss: {:.6f}".format(
                epoch+1, time.time()-start_time, d0_loss, d0_acc, d1_loss, d1_acc, g_loss))
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
            self._plot_loss()
            # save generated sequence every 100 epochs
            if epoch % sample_rate == 0:
                # save generated result of adversarial train
                result = generated_examples_batch.transpose(
                    0, 1).cpu().detach().numpy()
                rescaled_result = np.array(list(map(dataset.rescale, result)))
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
        self.pretrain_optimizer.zero_grad()
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
        self.pretrain_optimizer.step()
        return loss.item()

    def _sample_pos_neg_examples(self, dataset, sample_size):
        real_examples_batch = self._sample_examples_batch(
            dataset, sample_size).transpose(0, 1).to(device)
        first_motion_input = real_examples_batch[0]
        latent_code_batch = self._sample_latent_code_batch(
            sample_size, self.max_len, self.latent_code_size).transpose(0, 1).to(device)
        init_hidden = self.generator.zero_hidden_state(
            sample_size).to(device)
        generated_examples_batch, _ = self.generate_sequence(
            first_motion_input, latent_code_batch, init_hidden, teacher_sequence=None)
        return real_examples_batch, generated_examples_batch

    def _sample_examples_batch(self, dataset, sample_size):
        """Randomly choose number of batch_size samples from dataset"""
        sampled_idxs = np.random.choice(len(dataset), sample_size)
        sampled_examples = torch.cat(
            [dataset[i].unsqueeze(0) for i in sampled_idxs])
        return sampled_examples

    def _sample_latent_code_batch(self, sample_size, time_step, latent_code_size):
        """Normal distribution random value"""
        return torch.randn(sample_size, time_step, latent_code_size)

    def _plot_loss(self):
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
        ax2.set_ylim([0, 1])
        ax2.grid()
        ax2.legend()
        plt.tight_layout()
        plt.savefig('curve.png')
        plt.close()
