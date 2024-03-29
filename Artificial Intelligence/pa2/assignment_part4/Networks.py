import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=5, outptut_size=1):
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.nonlinear_activation = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(hidden_size, outptut_size)

    # STUDENTS: __init__() must initiatize nn.Module and define your network's
    # custom architecture

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        output = self.hidden_to_output(hidden)
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        loss = 0
        for idx, sample in enumerate(test_loader):
            loss += loss_function(self.forward(sample['input']), sample['label'])
        result = loss / len(test_loader)
        return result


def main():
    model = Action_Conditioned_FF()


if __name__ == '__main__':
    main()
