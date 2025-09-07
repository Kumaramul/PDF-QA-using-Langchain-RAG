import tensorflow_hub as hub
import numpy as np
from langchain.embeddings.base import Embeddings


class USEEmbeddings(Embeddings):
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model(texts).numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model([text]).numpy()[0].tolist()

