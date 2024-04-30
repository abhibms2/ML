#Transformer (attention) with enocder and decoder based.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten


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


def transformer_decoder_layer(d_model, num_heads, dff, rate=0.1):
    def layer(x, enc_output):
        attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x, x)
        attn1 = Dropout(rate)(attn1)
        out1 = LayerNormalization(epsilon=1e-6)(attn1 + x)

        attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(out1, enc_output, enc_output)
        attn2 = Dropout(rate)(attn2)
        out2 = LayerNormalization(epsilon=1e-6)(attn2 + out1)

        ffn_output = Dense(dff, activation='relu')(out2)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(ffn_output + out2)
    return layer


def transformer_encoder(inputs, num_layers, d_model, num_heads, dff, rate=0.1):
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_layer(d_model, num_heads, dff, rate)(x)
    return x


def transformer_decoder(inputs, enc_output, num_layers, d_model, num_heads, dff, rate=0.1):
    x = inputs
    for _ in range(num_layers):
        x = transformer_decoder_layer(d_model, num_heads, dff, rate)(x, enc_output)
    return x


# Configuration for the Transformer layers
d_model = 512
num_heads = 8
dff = 2048
num_layers = 4


input_tensor = Input(shape=(15973, 1))  # Adjust the input shape to include feature size
# Expand feature dimension to match d_model using a Dense layer
input_tensor_d_model = Dense(d_model)(input_tensor)


target_tensor = Input(shape=(1124012, 1))  # Input shape for decoder; adjust as needed
target_tensor_d_model = Dense(d_model)(target_tensor)  # Prepare target for processing by expanding features


encoder_output = transformer_encoder(input_tensor_d_model, num_layers, d_model, num_heads, dff)
decoder_output = transformer_decoder(target_tensor_d_model, encoder_output, num_layers, d_model, num_heads, dff)


# Assuming the output needs to be flattened or processed to match Y_train shape
decoder_output_processed = Flatten()(decoder_output)
output_tensor = Dense(1124012)(decoder_output_processed)  # Adjust this layer to ensure output matches Y_train dimensions


model = Model(inputs=[input_tensor, target_tensor], outputs=output_tensor)
model.compile(optimizer='adam', loss='mse')
model.summary()


# Assuming X_train and Y_train are prepared and correctly shaped
history = model.fit(
    [X_train, Y_train],  # input data
    Y_train,             # target data
    batch_size=1,        # batch size
    epochs=10,           # number of epochs
    verbose=1,           # verbosity level
    validation_split=0.2 # split percentage for validation data
)
