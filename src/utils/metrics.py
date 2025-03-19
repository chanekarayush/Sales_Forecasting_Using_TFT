import tensorflow as tf
from tensorflow.keras import layers

def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.MultiHeadAttention(num_heads=4, key_dim=2)(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    
    # Decoder
    x = layers.MultiHeadAttention(num_heads=4, key_dim=2)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)  # Output layer for regression
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Example input shape (timesteps, features)
input_shape = (30, num_features)  # Adjust based on your data
model = create_transformer_model(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')