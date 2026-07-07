import json
import sys
from pathlib import Path

from groq import Groq

from config import MODERATOR_MODEL_NAME

MODERATOR_PROMPT_PATH = Path(__file__).parent / "prompts" / "moderator_system_prompt.txt"


class Moderator:
    """Agent dédié : décide si une question est une tentative de prompt injection.

    Confier cette décision à un modèle séparé (plutôt qu'à une consigne ajoutée
    au prompt du RAG) évite qu'une injection réussie dans le prompt principal
    compromette à la fois la détection et la réponse : les deux rôles sont
    isolés, et le RAG ne voit jamais la question tant qu'elle n'est pas validée.
    """

    def __init__(self, client: Groq):
        self.client = client
        self.system_prompt = MODERATOR_PROMPT_PATH.read_text(encoding="utf-8")

    def moderate(self, question: str) -> dict:
        # Fail-safe : si le modérateur plante ou renvoie du JSON invalide, on
        # traite la question comme une injection potentielle plutôt que de
        # laisser passer une question non vérifiée ou de faire planter le pipeline.
        try:
            response = self.client.chat.completions.create(
                model=MODERATOR_MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Moderator] échec de la modération, fail-safe activé : {e!r}", file=sys.stderr)
            return {"is_prompt_injection": True}
