import numpy as np
import re
import urllib.parse
import pandas as pd
from gensim.models import Word2Vec
from sklearn.utils import resample

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
for input_url in X_resampled:
    standard_url = standardize_url(input_url)
    url_tokens = tokenize_url(standard_url)
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]
    if url_vectors:
        avg_vector = np.mean(url_vectors, axis=0)
    else:
        avg_vector = np.zeros(model.vector_size)

    url_vectors_list.append(avg_vector)

embeddings_spam_url = np.array(url_vectors_list)

Spam_Embedding_by_Word2vec = np.hstack((embeddings_spam_url, y_resampled.reshape([-1,1])))
pd.DataFrame(Spam_Embedding_by_Word2vec).to_csv('Spam_Embedding_by_Semantic_Only.csv', index=False)

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

for input_url in X_resampled:
    standard_url = standardize_url(input_url)
    url_tokens = tokenize_url(standard_url)
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]

    if url_vectors:
        avg_vector = np.mean(url_vectors, axis=0)
    else:
        avg_vector = np.zeros(model.vector_size)

    url_vectors_list.append(avg_vector)

embeddings_News_url = np.array(url_vectors_list)
News_Embedding_by_Word2Vec = np.hstack((embeddings_News_url, np.array(y_resampled).reshape([-1,1])))
pd.DataFrame(News_Embedding_by_Word2Vec).to_csv('News_Embedding_by_Semantic_Only.csv', index=False)

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

for input_url in X_resampled:
    standard_url = standardize_url(input_url)
    url_tokens = tokenize_url(standard_url)
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]

    if url_vectors:
        avg_vector = np.mean(url_vectors, axis=0)
    else:
        avg_vector = np.zeros(model.vector_size)

    url_vectors_list.append(avg_vector)

embeddings_malicious_phish_url = np.array(url_vectors_list)
malicious_phish_Embedding_by_Word2Vec = np.hstack((embeddings_malicious_phish_url, np.array(y_resampled).reshape([-1,1])))
pd.DataFrame(malicious_phish_Embedding_by_Word2Vec).to_csv('Malicious_Phish_Embedding_by_Semantic_Only.csv', index=False)


# classification
Classification_url = pd.read_csv('../../../Dataset/URL_C.csv')['URL'].values
label = pd.read_csv('../../../Dataset/URL_C.csv')['Type'].values
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

for input_url in X_resampled:
    standard_url = standardize_url(input_url)
    url_tokens = tokenize_url(standard_url)
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]

    if url_vectors:
        avg_vector = np.mean(url_vectors, axis=0)
    else:
        avg_vector = np.zeros(model.vector_size)

    url_vectors_list.append(avg_vector)

embeddings_classification_url = np.array(url_vectors_list)
Classification_Embedding_by_Word2vec = np.hstack((embeddings_classification_url, y_resampled.reshape([-1,1])))
pd.DataFrame(Classification_Embedding_by_Word2vec).to_csv('Classification_Embedding_by_Semantic_Only.csv', index=False)

# app
App_url = pd.read_csv('../../../Dataset/URL_A.csv')['URL'].values
label = pd.read_csv('../../../Dataset/URL_A.csv')['App'].values

data_train = [tokenize_url(line.strip()) for line in App_url]
model = Word2Vec(data_train, vector_size=10, window=5, min_count=1, sg=0)
url_vectors_list = []

for input_url in App_url:
    standard_url = standardize_url(input_url)
    url_tokens = tokenize_url(standard_url)
    url_vectors = [model.wv[token] for token in url_tokens if token in model.wv]

    if url_vectors:
        avg_vector = np.mean(url_vectors, axis=0)
    else:
        avg_vector = np.zeros(model.vector_size)

    url_vectors_list.append(avg_vector)

embeddings_App_url = np.array(url_vectors_list)
App_Embedding_by_Word2vec = np.hstack((embeddings_App_url, label.reshape([-1,1])))
pd.DataFrame(App_Embedding_by_Word2vec).to_csv('App_Embedding_by_Semantic_Only.csv', index=False)
