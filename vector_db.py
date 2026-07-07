import uuid
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME


class VectorDB:
    """Base vectorielle persistante (ChromaDB + sentence-transformers).

    - Si une collection existe déjà à `path`, elle est rechargée (y compris le
      modèle d'embedding utilisé à sa création, lu dans les métadonnées de la
      collection).
    - Sinon, si `chunks` est fourni, la collection est créée et indexée.
    - Sinon, erreur explicite : impossible de démarrer sans base ni chunks.
    """

    def __init__(
        self,
        path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        chunks: Optional[List[str]] = None,
        source: str = "corpus",
    ):
        self.client = chromadb.PersistentClient(path=path)

        try:
            self.collection = self.client.get_collection(collection_name)
            embedding_model_name = self.collection.metadata["embedding_model"]
            self.model = SentenceTransformer(embedding_model_name)
        except Exception:
            if not chunks:
                raise ValueError(
                    f"Aucune base ChromaDB existante à '{path}' pour la collection "
                    f"'{collection_name}', et aucun chunk n'a été fourni pour en créer une nouvelle."
                )

            embedding_model_name = EMBEDDING_MODEL_NAME
            self.model = SentenceTransformer(embedding_model_name)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"embedding_model": embedding_model_name},
            )

            embeddings = self._encode(chunks, show_progress_bar=True)
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": source} for _ in chunks]
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )

    def _encode(self, texts: List[str], show_progress_bar: bool = False):
        # normalize_embeddings=True : les vecteurs sont mis sur la sphère unité,
        # ce qui rend le produit scalaire équivalent à la similarité cosinus
        # utilisée par ChromaDB pour le classement des résultats.
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

    def retrieve(self, question: str, n: int = 3) -> dict:
        query_embedding = self._encode([question])
        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
