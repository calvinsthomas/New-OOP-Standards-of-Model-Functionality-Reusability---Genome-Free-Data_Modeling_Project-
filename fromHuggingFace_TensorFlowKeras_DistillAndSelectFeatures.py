#!/usr/bin/env python
# coding: utf-8

# In[ ]:

## RESULT (BELOW): LESS THAN 4 PROMPTS ONLY IN HUGGINGFACE's EXECUTIVE CHAT (BUILT-IN CHAT OPTION: CLONE OF HUGGINGFACE CTO)


import tensorflow as tf
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Tuple

def distill_and_select_feature(X: np.ndarray, y: np.ndarray, num_layers: int, hidden_size: int, activation: str='relu') -> Tuple[float, float]:
    r"""
    Trains a fully connected network with specified number of layers and size, fits it to each column of the input dataset,
    calculates R^2 scores, and returns the highest scoring input column's indices and corresponding R^2 value.

    Arguments:
        X {np.ndarray} -- Input dataset of shape (n_samples, n_features).
        y {np.ndarray} -- Target dataset of length n_samples.
        num_layers {int} -- Number of fully connected layers in the trained model.
        hidden_size {int} -- Size of hidden units for each fully connected layer.
        activation {str} -- Activation function used throughout the model (default: 'relu').
            
    Returns:
        Tuple[float, float] -- Highest scoring input column's index and corresponding R^2 value.
    """
    def build_tf_model(num_inputs: int, num_outputs: int, hidden_units: int, num_layers: int, activation: str) -> nn.Sequential:
        """
        Builds a fully connected model using TensorFlow's keras API.
        :param num_inputs: Int, number of inputs.
        :param num_outputs: Int, number of outputs.
        :param hidden_units: Int, size of hidden units for each fully connected layer.
        :param num_layers: Int, number of fully connected layers.
        :param activation: Str, activation function used throughout the model.
        :return: keras Model instance.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_units, input_dim=num_inputs, activation=activation),
        ])

        for i in range(1, num_layers):
            model.add(tf.keras.layers.Dense(hidden_units, activation=activation))

        model.add(tf.keras.layers.Dense(num_outputs))

        return model

    def build_pytorch_model(num_inputs: int, num_outputs: int, hidden_units: int, num_layers: int, activation: str) -> nn.Module:
        """
        Builds a fully connected model using Pytorch framework.
        :param num_inputs: Int, number of inputs.
        :param num_outputs: Int, number of outputs.
        :param hidden_units: Int, size of hidden units for each fully connected layer.
        :param param num_layers: Int, number of fully connected layers.
        :param activation: Str, activation Function used throughout the model.
        :return: Pytorch Module instance.
        """
        model = nn.Sequential(
            nn.Linear(num_inputs, hidden_units),
            nn.ReLU(),
        )

        for i in range(1, num_layers):
            model.add_module(f'fc_{i}', nn.Linear(hidden_units, hidden_units))
            model.add_module(f'act_{i}', nn.ReLU())

        model.add_module('output', nn.Linear(hidden_units, num_outputs))

        return model

    if isinstance(X, np.ndarray):
        X = tf.constant(X.astype('float32'))
        y = tf.constant(y.astype('float32'))
    elif isinstance(X, tf.Tensor):
        pass
    else:
        raise TypeError('Invalid input type, expected NumPy array or TensorFlow tensor.')

    # Create the dataset object for TensorFlow Dataset API
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y)).batch(len(X))

    # Initialize variables for tracking best R² score
    max_r2_val = -1
    chosen_column_index = 0

    # Loop over all columns to find the best fit
    for idx in range(X.shape[-1]):
        current_dataset = ds.map(lambda x: ({**{k: v[:, idx] for k, v in x.items()}, **{'target': x['target']}})
                                ).prefetch(buffer_size=tf.data.AUTOTUNE)

        if tf.__version__.startswith('2.'):
            model = build_tf_model(num_inputs=1, num_outputs=1, hidden_units=hidden_size, num_layers=num_layers, activation=activation)
            optimizer = tf.optimizers.Adam()
            loss_fn = tf.keras.losses.MeanSquaredError()

            @tf.function
            def train_step(_, x, y):
                nonlocal max_r2_val
                with tf.GradientTape() as tape:
                    predictions = model(x)
                    loss = loss_fn(y, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

return loss

max_r2_val, _, _, _ = evaluate_model(model, current_dataset)
else:
model = build_pytorch_model(num_inputs=1, num_outputs=1, hidden_units=hidden_size, num_layers=num_layers, activation=activation)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(current_dataset):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    eval_running_loss = running_loss / len(current_dataset.dataset)
    _, predicted = model.forward(next(iter(current_dataset)))
    max_r2_val, _, _, _ = calculate\_r2\_score(predicted.detach().numpy(), targets.numpy())

if tf.__version__.startswith('2.'):
    del model, optimizer, loss_fn
elif torch.__version__.startswith('1.'):
    del model, criterion, optimizer
gc.collect()

return max_r2_val, chosen_column_index

