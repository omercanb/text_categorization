import torch.nn as nn

class MLP_Model(nn.Module):
    def __init__(self, layers, units, dropout_rate, input_shape, num_classes):
        super(self, MLP_Model).__init__()
        output_units, activation = _get_last_layer_units_and_activation(num_classes)

        dropout1 = nn.Dropout(dropout_rate)
        network = []
        for layer in range(layers - 1):
            network.append(nn.Linear(units, units))
            network.append(nn.ReLU())
            network.append(nn.Dropout(dropout_rate))
        fc_final = nn.Linear(units, output_units)

        activation_function = nn.Softmax() if activation == "softmax" else nn.Sigmoid()


        self.fc_stack = nn.Sequential(dropout1, *layers, fc_final, activation_function)

    def forward(self, x):
        return self.fc_stack(x)
    

from keras import models
from keras.layers import Dense
from keras.layers import Dropout

def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def _get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation