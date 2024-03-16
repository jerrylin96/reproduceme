import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
import tensorflow_addons as tfa
from qhoptim.tf import QHAdamOptimizer
from tensorflow.keras import callbacks
import os
import random
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import logging

print('imported packages')

data_path = "/ocean/projects/atm200007p/jlin96/nnspreadtesting_2/overfitmepls/training/training_data/"
train_input = np.load(data_path + 'train_input.npy', mmap_mode='r')
train_target = np.load(data_path + 'train_target.npy', mmap_mode='r')
val_input = np.load(data_path + 'val_input.npy', mmap_mode='r')
val_target = np.load(data_path + 'val_target.npy', mmap_mode='r')

num_epochs = 2
project_name = 'my-first-sweep'

# Define sweep config
wandb.login()

print('logged into wandb')

def build_model(hp:dict):
    alpha = hp["leak"]
    dp_rate = hp["dropout"]
    batch_norm = hp["batch_normalization"]
    model = Sequential()
    hidden_units = hp['hidden_units']
    model.add(Dense(units = hidden_units, input_dim=175, kernel_initializer='normal'))
    model.add(LeakyReLU(alpha = alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dp_rate))
    for i in range(hp["num_layers"]):
        model.add(Dense(units = hidden_units, kernel_initializer='normal'))
        model.add(LeakyReLU(alpha = alpha))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dp_rate))
    model.add(Dense(55, kernel_initializer='normal', activation='linear'))
    initial_learning_rate = hp["learning_rate"]
    optimizer = hp["optimizer"]
    if optimizer == "adam":
        optimizer = keras.optimizers.Adam(learning_rate = initial_learning_rate)
    elif optimizer == "RAdam":
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate = initial_learning_rate)
    elif optimizer == "QHAdam":
        optimizer = QHAdamOptimizer(learning_rate = initial_learning_rate, nu2=1.0, beta1=0.995, beta2=0.999)
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ["mse"])
    return model

def data_generator(inputs, targets):
    # Iterate over the dataset in batches
    for i in range(0, len(inputs)):
        # Yield a batch of data
        yield inputs[i], targets[i]

def main():
    logging.basicConfig(level=logging.DEBUG)
    wandb.init(project=project_name)
    batch_size = wandb.config['batch_size']
    shuffle_buffer = wandb.config['shuffle_buffer']
    with tf.device('/CPU:0'):
        train_ds = tf.data.Dataset.from_generator(
            lambda: data_generator(train_input, train_target),
                                output_signature=(tf.TensorSpec(shape=(175), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(55), dtype=tf.float32))).shuffle(buffer_size=shuffle_buffer).batch(batch_size)

        val_ds = tf.data.Dataset.from_generator(
            lambda: data_generator(val_input, val_target),
                                output_signature=(tf.TensorSpec(shape=(175), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(55), dtype=tf.float32))).shuffle(buffer_size=shuffle_buffer).batch(batch_size)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    logging.debug("Data loaded")
    model = build_model(wandb.config)
    model.fit(train_ds, validation_data = val_ds, epochs = num_epochs, callbacks = [WandbMetricsLogger(), WandbModelCheckpoint('tuning_directory')])

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "batch_size": {"values": [5000, 10000, 20000]},
        "shuffle_buffer": {"values": [20000, 40000]},
        "leak": {"min": 0, "max": 0.4},
        "dropout": {"min": 0, "max": 0.25},
        "learning_rate": {'distribution': 'log_uniform_values', "min": 1e-6, "max": 1e-3},
        "num_layers": {'distribution': 'int_uniform', "min": 4, 'max': 11},
        "hidden_units": {'distribution': 'int_uniform', "min": 200, 'max': 480},
        "optimizer": {"values": ["adam", "RAdam", "QHAdam"]},
        "batch_normalization": {"values": [True, False]}
    }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=main, count=2)