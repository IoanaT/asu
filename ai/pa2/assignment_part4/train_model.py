from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    losses = []
    loss_function = nn.MSELoss()
    learning_rate = 0.1
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader):
            optimizer.zero_grad()
            output = model(sample['input'])
            loss = loss_function(output, sample['label'])
            loss.backward()
            optimizer.step()
        torch.serialization.save(model.state_dict(), "saved/saved_model.pkl")


if __name__ == '__main__':
    no_epochs = 10000
    train_model(no_epochs)
