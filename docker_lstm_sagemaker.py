import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def model_fn(input_shape, output_shape):
    model = Sequential([
        LSTM(1024, input_shape = input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(512, return_sequences=True),
        Dropout(0.2),
        LSTM(256, return_sequences=False),
        Dropout(0.2),
        Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

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

    input_shape = (15973, 1)

    output_shape = 1124012

    X_train = X_train.reshape((-1,) + input_shape)
    # Get the model
    model = model_fn(input_shape, output_shape)

    # Train the model
    model.fit(X_train, Y_train, batch_size=1, epochs=10, verbose=1)

    # Save the model to the location specified by SageMaker
    model_output_directory = os.path.join(args.model_dir, '1')  # Ensure the model is saved in a versioned directory
    tf.saved_model.save(model, model_output_directory)
