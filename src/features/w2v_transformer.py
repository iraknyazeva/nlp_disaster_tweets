from pathlib import Path
import sys

# добавляем корень проекта в sys.path для импортов из src 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
import numpy as np

# получаем эмбеддинги твитов усреднением эмбеддингов токенов, полученных word2vec
def get_w2v_texts(texts, w2v):
    embeddings = []
    for text in texts:
        tokens = text.split()
        vectors = [w2v.wv[token] for token in tokens if token in w2v.wv]
        if len(vectors) > 0:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(w2v.vector_size))
    return np.array(embeddings)

# кастомный класс, чтобы был совместим с sklearn и можно было в pipeline для logreb положить
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300, window=3, negative=10, min_count=2, seed=42):
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.min_count = min_count
        self.seed = seed
        self.w2v_ = None

    def fit(self, X, y=None):
        tokenized = [text.split() for text in X]
        self.w2v_ = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            negative=self.negative,
            min_count=self.min_count,
            seed=self.seed
        )
        return self

    def transform(self, X):
        return get_w2v_texts(X, self.w2v_)