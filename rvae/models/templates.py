import torch

from torch import nn
from ..geoml import nnj


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h

        
class MLP(nn.Module):
    def __init__(self, n_in, layer_defs, act, n_out, out_act=None):
        """
        A simple Multilayer Perceptron model.

        params:
            n_in:           integer - the input size
            layer_defs:     list of integers - the hidden layer sizes
            n_out:          integer - the output size
            act:            torch Module object - the activation functions
            out_act         torch Module object - the output activation function
                            By default the MLP will have linear output.
        """
        super(MLP, self).__init__()

        # take care of the input layer
        self.net = [nnj.Linear(n_in, layer_defs[0]), act()]
        self.linear_layers = [nnj.Linear(layer_defs[i], layer_defs[i + 1]) for i in range(len(layer_defs) - 1)]

        for linear in self.linear_layers:
            self.net.extend([
                linear,
                act()
            ])
        # append the output layers
        if n_out is None:
            pass
        else:
            if out_act is None:
                self.net.extend([nnj.Linear(layer_defs[-1], n_out)])
            else:
                self.net.extend([nnj.Linear(layer_defs[-1], n_out), out_act()])

        self.net = nnj.Sequential(*self.net)

    def forward(self, x, jacobian=None):
        return self.net(x, jacobian)
