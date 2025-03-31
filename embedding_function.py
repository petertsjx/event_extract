from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

model_path = "..\\text2vec-base-chinese"
model = SentenceTransformer(model_path,"D:/huggingface_cache/")

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        embeddings = [model.encode(x) for x in texts]
        return embeddings