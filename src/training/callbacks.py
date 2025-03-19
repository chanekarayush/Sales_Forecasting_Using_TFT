import tensorflow as tf
from tensorflow.keras import layers, models

def create_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Transformer block
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)  # Residual connection
    x = layers.Conv1D(filters=128, kernel_size=1, activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)  # Residual connection
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1)(x)  # Predicting the next order quantity
    
    model = models.Model(inputs, outputs)
    return model

# Example input shape (number of time steps, number of features)
input_shape = (12, num_features)  # 12 months of data
model = create_transformer_model(input_shape)
model.compile(optimizer='adam', loss='mse')