from abc import ABC, abstractmethod
from typing import List

# Data Object for passing text around


class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

# INTERFACE 1: Something that loads documents


class IDocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        pass

# INTERFACE 2: Something that turns text into vectors


class IEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

# INTERFACE 3: Something that stores vectors


class IVectorStore(ABC):
    @abstractmethod
    def upsert(self, documents: List[Document], embeddings: List[List[float]]):
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Document]:
        pass
