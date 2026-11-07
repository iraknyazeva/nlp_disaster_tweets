from sentence_transformers import SentenceTransformer
import numpy as np

def get_embeddings(texts, model_name, batch_size, show_progress_bar):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
      texts,
      batch_size=batch_size,
      show_progress_bar=show_progress_bar,
      convert_to_numpy=True
    )
    return embeddings
    