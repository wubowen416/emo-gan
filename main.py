import pickle
import numpy as np

from dataloader import MyDataset
from gan import GAN


if __name__ == "__main__":

    # info of dataset
    num_joints = 3
    dim = 10

    # dataset config
    max_len = 300  # max length of sequence used in training

    # network config
    latent_code_size = 10  # dim of latent code
    hidden_size = 128
    generator_output_size = 30  # motion data
    discriminator_output_size = 1  # binary classification
    num_layers = 2
    bidirectional = True  # discriminator direction
    relu_slope = 1e-2  # leaky relu

    # training config
    epochs = 500
    batch_size = 16  # mini-batch size
    sample_size = 128  # number of samples to train d&g over one epoch
    learning_rate = {'pretrain': 1e-5, 'g': 1e-3, 'd': 1e-3}
    k_step = 1  # step of training discriminator over one epoch
    sample_rate = 50 # save result every sample_reate epochs
    dropout = 0
    pretrain_epochs = 30 # 0 is not to do pretrain
    teacher_forcing_rate = 0.7

    save_path = '/home/wu/projects/emo-gan/chkpt/rnngan'
    # load data
    train_data_path = "/home/wu/mounts/Emo-gesture/train_set.pkl"
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = MyDataset(data, max_len=max_len, num_joints=3, dim=dim)

    gan = GAN(latent_code_size, hidden_size, generator_output_size, discriminator_output_size,
              num_layers, bidirectional, relu_slope, dropout, max_len=300)
    gan.fit(dataset, epochs, batch_size, sample_size, learning_rate,
            k_step, sample_rate, pretrain_epochs, teacher_forcing_rate, save_path)
