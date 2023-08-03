import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from training_and_testing.utils import build_module

# TODO: Here you should create your models. You can use the MLPModel or ConstantVelocity as a template.
#  Each model should have a __init__ function, a forward function, and a loss_function function.
#  The loss function doen't have to be in the model, but it is convenient to have it there, because the lit_module
#  will call it automatically, because you assign a prediction model to it and later it asks the model for the loss function.

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()

        self.backbone = nn.Linear(input_size, hidden_size)
        self.layers = build_module(
            hidden_size, output_size, num_layers, self.backbone)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def loss_function(self, y, y_hat):
        y_view = y.view(y.size(0), -1)
        loss = F.mse_loss(y_hat, y_view)
        return loss


class ConstantVelocity(nn.Module):
    def __init__(self, sequence_length, past_sequence_length, future_sequence_length):
        super().__init__()
        self.future_sequence_length = future_sequence_length
        self.past_sequence_length = past_sequence_length
        self.sequence_length = sequence_length
        self.layers = nn.Identity()

    def forward(self, x):
        # TODO: Here we select columns of the data. If you change your dataset/features, you should change this as well.
        positions = x[:, -1, 4:6]
        velocities = x[:, -1, 9:11]
        predictions = []
        for t in range(0, self.future_sequence_length):
            positions = positions + velocities
            positions_squeezed = positions.unsqueeze(1)
            predictions.append(positions_squeezed)
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def loss_function(self, y, y_hat):
        # TODO: Here we select columns of the data. If you change your dataset/features, you should change this as well.
        positions_y = y[:, :, 4:6]
        positions_y_hat = y_hat[:, :, :]
        # velocities_y = y[:, :, 9:11]
        # velocities_y_hat = y_hat[:, :, 9:11]
        loss = F.mse_loss(positions_y, positions_y_hat)
        return loss
