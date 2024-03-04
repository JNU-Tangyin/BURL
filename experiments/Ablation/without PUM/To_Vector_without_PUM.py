import pandas as pd
from tensorflow.keras.models import load_model, Model
import urllib.parse
from tqdm import tqdm
from sklearn.utils import resample
from gensim.models import Word2Vec
import numpy as np
import urllib.parse
import re

# Loading Model
BURL_without_PUM = load_model('../../../Save_Model/BURL_without_PUM.h5')


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

def url_to_vector(url):
    url_tokens = tokenize_url(standardize_url(url))
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]
    input_dim = 10
    if len(url_vectors) < input_dim:
        padding = [np.zeros(model.vector_size)] * (input_dim - len(url_vectors))
        url_vectors += padding
    elif len(url_vectors) > input_dim:
        url_vectors = url_vectors[:input_dim]
    url_vectors_array = np.array([url_vectors])
    return url_vectors_array


# spam
spam_url = pd.read_csv('../../../Dataset/S_URL.csv')['URL'].values
label = pd.read_csv('../../../Dataset/S_URL.csv')['is_spam'].values
n_classes = len(np.unique(label))
X_resampled = []
y_resampled = []

for c in np.unique(label):
    idx = label == c
    true_indices = np.where(idx)[0]

    X_class = spam_url[true_indices]
    y_class = label[true_indices]
    X_class_resampled, y_class_resampled = resample(X_class, y_class, n_samples=1000, random_state=42)
    X_resampled.append(X_class_resampled)
    y_resampled.append(y_class_resampled)

X_resampled = np.concatenate((X_resampled[0], X_resampled[1]))
y_resampled = np.concatenate((y_resampled[0], y_resampled[1]))
data_train = [tokenize_url(line.strip()) for line in X_resampled]
model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)
url_vectors_list = []

for url in X_resampled:
    url_vectors_list.append(url_to_vector(url))

compressed_representation = []
encoder = Model(BURL_without_PUM.input, BURL_without_PUM.layers[-3].output)
for url_vector in tqdm(url_vectors_list, desc='Processing'):
    compressed_representation.append(encoder.predict(url_vector))

embeddings_App_url = np.array(compressed_representation)[:, 0, :]
App_Embedding_by_SentenceBert = np.hstack((embeddings_App_url, y_resampled.reshape([-1,1])))
pd.DataFrame(App_Embedding_by_SentenceBert).to_csv('Spam_Embedding_by_Model_without_PUM.csv', index=False)

# News
News_url = pd.read_csv('../../../Dataset/NPJ.csv', encoding = 'ISO-8859-1')['url'].values
label = pd.read_csv('../../../Dataset/NPJ.csv', encoding = 'ISO-8859-1')['top_article'].values
n_classes = len(np.unique(label))
X_resampled = []
y_resampled = []

for c in np.unique(label):
    idx = label == c
    true_indices = np.where(idx)[0]

    X_class = News_url[true_indices]
    y_class = label[true_indices]

    X_class_resampled, y_class_resampled = resample(X_class, y_class, n_samples=1000, random_state=42)
    X_resampled.append(X_class_resampled)
    y_resampled.append(y_class_resampled)

X_resampled = np.concatenate((X_resampled[0], X_resampled[1]))
y_resampled = np.concatenate((y_resampled[0], y_resampled[1]))
data_train = [tokenize_url(line.strip()) for line in X_resampled]

model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)

url_vectors_list = []

for url in X_resampled:
    url_vectors_list.append(url_to_vector(url))

compressed_representation = []
encoder = Model(BURL_without_PUM.input, BURL_without_PUM.layers[-3].output)
for url_vector in tqdm(url_vectors_list, desc='Processing'):
    compressed_representation.append(encoder.predict(url_vector))
embeddings_News_url = np.array(compressed_representation)[:, 0, :]
News_Embedding_by_SentenceBert = np.hstack((embeddings_News_url, y_resampled.reshape([-1,1])))
pd.DataFrame(News_Embedding_by_SentenceBert).to_csv('News_Embedding_by_Model_without_PUM.csv', index=False)

# malicious
malicious_phish_url = pd.read_csv('../../../Dataset/M_URL.csv', encoding = 'ISO-8859-1')['URL'].values
label = pd.read_csv('../../../Dataset/M_URL.csv', encoding = 'ISO-8859-1')['type'].values
n_classes = len(np.unique(label))
X_resampled = []
y_resampled = []

for c in np.unique(label):
    idx = label == c
    true_indices = np.where(idx)[0]

    X_class = malicious_phish_url[true_indices]
    y_class = label[true_indices]

    X_class_resampled, y_class_resampled = resample(X_class, y_class, n_samples=1000, random_state=42)
    X_resampled.append(X_class_resampled)
    y_resampled.append(y_class_resampled)

X_resampled = np.concatenate((X_resampled[0], X_resampled[1], X_resampled[2], X_resampled[3]))
y_resampled = np.concatenate((y_resampled[0], y_resampled[1], y_resampled[2], y_resampled[3]))
data_train = [tokenize_url(line.strip()) for line in X_resampled]

model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)

url_vectors_list = []

for url in X_resampled:
    url_vectors_list.append(url_to_vector(url))

compressed_representation = []
encoder = Model(BURL_without_PUM.input, BURL_without_PUM.layers[-3].output)
for url_vector in tqdm(url_vectors_list, desc='Processing'):
    compressed_representation.append(encoder.predict(url_vector))

embeddings_App_url = np.array(compressed_representation)[:, 0, :]
App_Embedding_by_SentenceBert = np.hstack((embeddings_App_url, y_resampled.reshape([-1,1])))
pd.DataFrame(App_Embedding_by_SentenceBert).to_csv('Malicious_Phish_Embedding_by_Model_without_PUM.csv', index=False)


# classification
Classification_url = pd.read_csv('../../../Dataset/URL_C')['URL'].values
label = pd.read_csv('../../../Dataset/URL_C')['Type'].values
n_classes = len(np.unique(label))
X_resampled = []
y_resampled = []

for c in np.unique(label):
    idx = label == c
    true_indices = np.where(idx)[0]

    X_class = Classification_url[true_indices]
    y_class = label[true_indices]

    X_class_resampled, y_class_resampled = resample(X_class, y_class, n_samples=1000, random_state=42)
    X_resampled.append(X_class_resampled)
    y_resampled.append(y_class_resampled)

X_resampled = np.concatenate((X_resampled[0], X_resampled[1], X_resampled[2], X_resampled[3], X_resampled[4], X_resampled[5], X_resampled[6], X_resampled[7], X_resampled[8], X_resampled[9], X_resampled[10], X_resampled[11]))
y_resampled = np.concatenate((y_resampled[0], y_resampled[1], y_resampled[2], y_resampled[3], y_resampled[4], y_resampled[5], y_resampled[6], y_resampled[7], y_resampled[8], y_resampled[9], y_resampled[10], y_resampled[11]))
data_train = [tokenize_url(line.strip()) for line in X_resampled]

model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)

url_vectors_list = []

for url in X_resampled:
    url_vectors_list.append(url_to_vector(url))

compressed_representation = []
encoder = Model(BURL_without_PUM.input, BURL_without_PUM.layers[-3].output)
for url_vector in tqdm(url_vectors_list, desc='Processing'):
    compressed_representation.append(encoder.predict(url_vector))

embeddings_App_url = np.array(compressed_representation)[:, 0, :]
App_Embedding_by_SentenceBert = np.hstack((embeddings_App_url, y_resampled.reshape([-1,1])))
pd.DataFrame(App_Embedding_by_SentenceBert).to_csv('Classification_Embedding_by_Model_without_PUM.csv', index=False)

# app
App_url = pd.read_csv('../../../Dataset/URL-A')['URL'].values
label = pd.read_csv('../../../Dataset/URL-A')['App'].values

data_train = [tokenize_url(line.strip()) for line in App_url]

model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)
url_vectors_list = []

for url in App_url:
    url_vectors_list.append(url_to_vector(url))


compressed_representation = []
encoder = Model(BURL_without_PUM.input, BURL_without_PUM.layers[-3].output)
for url_vector in tqdm(url_vectors_list, desc='Processing'):
    compressed_representation.append(encoder.predict(url_vector))

embeddings_App_url = np.array(compressed_representation)[:, 0, :]
App_Embedding_by_SentenceBert = np.hstack((embeddings_App_url, label.reshape([-1,1])))
pd.DataFrame(App_Embedding_by_SentenceBert).to_csv('App_Embedding_by_Model_without_PUM.csv', index=False)
