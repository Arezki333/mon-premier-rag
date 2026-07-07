import csv
from pathlib import Path

CORPUS_CSV_PATH = Path(__file__).parent / "05_corpus_rag.csv"


def load_corpus(path: Path = CORPUS_CSV_PATH) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "source": row["source"],
                "categorie": row["categorie"],
            }
            for row in reader
        ]
