from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Layer
from utils import *
def run(file_path, latent_dim, epochs, optimizer, loss, save_path):
    # Prepare Data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    # Semantic Information Capture
    model, url_vectors_array = Semantic_Capture(data, vector_size=10, window=5, min_count=1, sg=0)

    # Structure Information capture
    timesteps = url_vectors_array.shape[1]  # URL Length
    input_dim = url_vectors_array.shape[2]   # Input Size

    # Auto Encoder-Decoder
    # Encoder Layer
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim, return_sequences=True)(inputs)

    # Attention Layer
    attention = AttentionLayer()(encoded)
    repeated_attention = RepeatVector(timesteps)(attention)

    # Decoder Layer
    decoded = LSTM(input_dim, return_sequences=True)(repeated_attention)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.fit(url_vectors_array, url_vectors_array, epochs=epochs)
    encoder = Model(inputs, encoded)

    # Save File
    autoencoder.save(save_path)

if __name__ == '__main__':
    file_path = 'Training_Data.txt'
    latent_dim = 100
    epochs = 50
    optimizer = 'adam'
    loss = 'mse'
    save_path = 'Pretrain_model.h5'
    run(file_path, latent_dim, epochs, optimizer, loss, save_path)