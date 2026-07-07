from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from config import CHROMA_DB_PATH, COLLECTION_NAME, LLM_MODEL_NAME, N_RESULTS
from moderator import Moderator
from vector_db import VectorDB

RAG_PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_system_prompt.txt"

REFUSAL_MESSAGE = (
    "Je ne peux pas traiter cette demande : elle a été identifiée comme une "
    "tentative de détournement des instructions du système."
)


class RAG:
    def __init__(self, chunks: list[dict] | None = None):
        load_dotenv()

        self.client = Groq()
        self.moderator = Moderator(self.client)
        self.vector_db = VectorDB(
            path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            chunks=chunks,
        )
        self.system_prompt_template = RAG_PROMPT_PATH.read_text(encoding="utf-8")

    def _build_system_prompt(self, question: str) -> str:
        results = self.vector_db.retrieve(question, n=N_RESULTS)
        chunks = results["documents"][0]
        formatted_chunks = "\n".join(f"{i + 1}. {chunk}" for i, chunk in enumerate(chunks))
        return self.system_prompt_template.replace("{{Chunks}}", formatted_chunks)

    def answer_question(self, question: str) -> str:
        # La modération passe avant tout appel au LLM principal : si la
        # question est une injection, on ne la laisse jamais atteindre le
        # prompt système du RAG.
        moderation = self.moderator.moderate(question)
        if moderation.get("is_prompt_injection"):
            return REFUSAL_MESSAGE

        system_prompt = self._build_system_prompt(question)
        response = self.client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
