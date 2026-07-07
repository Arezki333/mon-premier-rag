"""Compare deux modèles d'embedding multilingues sur les mêmes questions de test.

Utilise un client ChromaDB éphémère (en mémoire) pour ne pas toucher à la base
persistée du projet (chroma_db/).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from data.corpus import load_corpus

MODELS_TO_COMPARE = [
    "distiluse-base-multilingual-cased-v2",
    "paraphrase-multilingual-mpnet-base-v2",
]

TEST_QUESTIONS = [
    "Quelle est la couleur du chat de Bob ?",
    "Quel est le sport officiel de Basse-Molette ?",
    "Combien de boutons de nacre Bob possède-t-il ?",
    "Quelle est la capitale du Japon ?",
    "Quand est né Bob ?",
]


def index_corpus(model_name: str, chunks: list[dict]):
    model = SentenceTransformer(model_name)
    collection = chromadb.EphemeralClient().get_or_create_collection(
        name="compare_" + model_name.replace("/", "_")
    )
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": chunk["source"], "categorie": chunk["categorie"]} for chunk in chunks],
    )
    return model, collection


def retrieve(model: SentenceTransformer, collection, question: str, n: int = 3):
    query_embedding = model.encode([question], normalize_embeddings=True)
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n,
        include=["documents", "distances"],
    )


if __name__ == "__main__":
    chunks = load_corpus()

    indexed = {}
    for model_name in MODELS_TO_COMPARE:
        print(f"Indexation avec {model_name}...")
        indexed[model_name] = index_corpus(model_name, chunks)

    for question in TEST_QUESTIONS:
        print(f"\n=== Q: {question} ===")
        for model_name in MODELS_TO_COMPARE:
            model, collection = indexed[model_name]
            results = retrieve(model, collection, question)
            print(f"-- {model_name} --")
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                print(f"  {dist:.4f}  {doc}")
