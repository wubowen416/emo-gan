import pickle

from dataloader import MyDataset
from gan import GAN


if __name__ == "__main__":

    # info of dataset
    num_joints = 3
    dim = 10

    # dataset config
    max_len = 300   # max length of sequence used in training
    

    # network config
    latent_code_size = 10
    hidden_size = 128
    generator_output_size = 30
    discriminator_output_size = 1
    num_layers = 1
    bidirectional = False

    # training config
    epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    k_step = 1
    dropout = 0
    patience = 20
    test_size = 0.2
    pretrain_epochs = 20
    teacher_forcing_rate = 0.7


    
    train_data_path = "/home/wu/mounts/Emo-gesture/train_set.pkl"
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)

    dataset = MyDataset(data, max_len=max_len, num_joints=3, dim=dim)

    gan = GAN(latent_code_size, hidden_size, generator_output_size, discriminator_output_size, num_layers, bidirectional, dropout, max_len=300)
    
    gan.fit(dataset, epochs, batch_size, learning_rate, k_step, patience, test_size, pretrain_epochs, teacher_forcing_rate)

