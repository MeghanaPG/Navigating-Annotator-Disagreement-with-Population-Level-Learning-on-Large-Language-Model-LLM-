import numpy as np
import tensorflow as tf

# bert_embeddings = np.load('bert_embeddings_file.npy')
def create_cnn_model(num_classes, bert_embeddings):
    # Define the model architecture
    model = tf.keras.Sequential([
        # Input layer
        # tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.InputLayer(input_shape=bert_embeddings.shape[1:]),
        # Flatten layer (if necessary)
        tf.keras.layers.Flatten(),
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Compiling the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
# model.summary()
