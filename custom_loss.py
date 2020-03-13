import torch
import torch.nn as nn


class DLoss(nn.Module):
    def __init__(self):
        super(DLoss, self).__init__()
        
    def forward(self, positive_output_batch, negative_output_batch):
        batch_size = positive_output_batch.size(0)
        # for positive examples
        p_loss = - torch.mean(torch.log(positive_output_batch))
        n_loss = - torch.mean(torch.log((1 - negative_output_batch)))
        return p_loss + n_loss, p_loss, n_loss
        

class GLoss(nn.Module):
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, d_output_batch):
        loss = - torch.mean(torch.log(d_output_batch))
        return loss