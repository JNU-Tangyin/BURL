import numpy as np
import urllib.parse
import re
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Layer
import tensorflow.keras.backend as K
from gensim.models import Word2Vec
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)


def standardize_url(url):
    if url.startswith("http://") or url.startswith("https://"):
        url = url.split("://")[1]
    return url.lower()

def tokenize_url(url):
    parsed_url = urllib.parse.urlparse(url)
    tokens = []
    separators = r"[.#=\+\$%\~:\?\&\-\_!\(\)\"\'\\\/\*;|{}@\s,]"

    domain_tokens = re.split(separators, parsed_url.netloc)
    tokens.extend(filter(None, domain_tokens))

    path_tokens = re.split(separators, parsed_url.path)
    tokens.extend(filter(None, path_tokens))

    query_tokens = re.split(separators, parsed_url.query)
    tokens.extend(filter(None, query_tokens))

    return tokens

def preprocess_data_for_word2vec(tokenized_data, context_window):
    # Flatten the list of token lists into a single list of tokens
    all_tokens = [token for sublist in tokenized_data for token in sublist]

    # Create a vocabulary set and dictionaries
    vocab = set(all_tokens)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    training_pairs = []
    for token_list in tokenized_data:
        for i, token in enumerate(token_list):
            token_id = word_to_id[token]
            context_ids = [word_to_id[token_list[j]] for j in range(max(0, i - context_window), min(len(token_list), i + context_window + 1)) if j != i]
            for context_id in context_ids:
                training_pairs.append((token_id, context_id))

    return np.array(training_pairs), word_to_id, id_to_word, len(vocab)
def Semantic_Capture(data, vector_size, window, min_count, sg):
    # Tokenize
    data_train = [tokenize_url(line.strip()) for line in data]
    # Semantic Information Capture
    model = Word2Vec(data_train, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    url_vectors_list = []

    for input_url in data:
        standard_url = standardize_url(input_url)
        url_tokens = tokenize_url(standard_url)
        url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]

        # Padding and Terminal
        input_dim = 10
        if len(url_vectors) < input_dim:
            padding = [np.zeros(model.vector_size)] * (input_dim - len(url_vectors))
            url_vectors += padding
        elif len(url_vectors) > input_dim:
            url_vectors = url_vectors[:input_dim]

        url_vectors_list.append(url_vectors)

    url_vectors_array = np.array(url_vectors_list)
    return model, url_vectors_array

