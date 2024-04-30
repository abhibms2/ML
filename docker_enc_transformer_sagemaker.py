import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten
from tensorflow.keras.callbacks import Callback

def transformer_encoder_layer(d_model, num_heads, dff, rate=0.1):
    def layer(x):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x, x)
        attn_output = Dropout(rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = Dense(dff, activation='relu')(out1)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return layer

def transformer_encoder(inputs, num_layers, d_model, num_heads, dff, rate=0.1):
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_layer(d_model, num_heads, dff, rate)(x)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    # Load data using the SageMaker environment variables
    train_data_dir = args.train  # Get the training data directory from SageMaker environment
    X_train_path = os.path.join(train_data_dir, 'X_train_2023-10-01_1_2023-10-02_23.npy')
    Y_train_path = os.path.join(train_data_dir, 'Y_train_2023-10-01_1_2023-10-02_23.npy')

    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)

    # Model parameters
    d_model = 256
    num_heads = 8
    dff = 1024
    num_layers = 3

    input_tensor = Input(shape=(15973, 1))  # Adjust the input shape to include feature size
    input_tensor_d_model = Dense(d_model)(input_tensor)  # Expand feature dimension to match d_model

    encoder_output = transformer_encoder(input_tensor_d_model, num_layers, d_model, num_heads, dff)

    # Flatten the encoder output to connect to a Dense layer for prediction
    encoder_output_flat = Flatten()(encoder_output)
    output_tensor = Dense(1124012, activation='linear')(encoder_output_flat)  # Adjust the output dimensions as needed for your task

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(optimizer='adam', loss='mse')#, metrics=['accuracy'])
    model.summary()

    # Train the model
    history = model.fit(
        X_train,  # input data
        Y_train,  # target data
        batch_size=1,  # batch size
        epochs=10,  # number of epochs
        verbose=1,  # verbosity level
        validation_split=0.2  # split percentage for validation data
    )

    # Save the model to the location specified by SageMaker
    model_output_directory = os.path.join(args.model_dir, '2')  # Ensure the model is saved in a versioned directory
    tf.saved_model.save(model, model_output_directory)

~
