
import os
import re
import unicodedata
from abc import ABC, abstractmethod

import requests


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(
        char for char in value
        if not unicodedata.combining(char)
    )


def is_french(question: str) -> bool:
    text = normalize_text(question)
    french_terms = [
        "quelle est", "combien de", "est-ce que",
        "puis-je", "je peux", "votre", "produit",
        "livraison", "remboursement", "politique de",
        "retourner", "comment",
    ]
    return any(term in text for term in french_terms)


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, question: str, context: str) -> str:
        pass


class RuleProvider(LLMProvider):

    def generate(self, question: str, context: str) -> str:
        text = normalize_text(question)
        source = normalize_text(context)
        french = is_french(question)

        return_terms = (
            "return", "refund", "retour",
            "remboursement", "echang"
        )

        if any(term in text for term in return_terms):
            # Only state facts explicitly supported by the
            # retrieved policy context.
            has_30_day_policy = (
                "30 days" in source
                or "30 jours" in source
            )
            has_conditions = (
                "unused" in source
                or "inutilise" in source
            )

            if has_30_day_policy:
                if french:
                    answer = (
                        "Vous pouvez retourner les produits "
                        "admissibles dans les 30 jours suivant "
                        "la livraison."
                    )
                    if has_conditions:
                        answer += (
                            " Le produit doit être inutilisé, "
                            "dans son emballage d'origine et "
                            "accompagné d'une preuve d'achat."
                        )
                else:
                    answer = (
                        "Eligible products can be returned "
                        "within 30 days of delivery."
                    )
                    if has_conditions:
                        answer += (
                            " Items must be unused, in their "
                            "original packaging, and accompanied "
                            "by proof of purchase."
                        )
                return answer

        if french:
            return (
                "Je ne peux pas confirmer une réponse précise "
                "à partir des informations récupérées. "
                "Veuillez contacter le service à la clientèle."
            )

        return (
            "I cannot confirm a precise answer from the "
            "retrieved information. Please contact "
            "customer service."
        )


class OpenAIProvider(LLMProvider):

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, question: str, context: str) -> str:
        if not self.api_key:
            return "OpenAI provider is not configured. OPENAI_API_KEY is missing."

        language = "French" if is_french(question) else "English"
        prompt = (
            "You are the Boréal Marché customer assistant. "
            "Answer only using the provided context. "
            "Do not invent policies or facts. "
            f"Answer in {language}. "
            "Keep the response concise. "
            "If the context does not support an answer, "
            "say so and recommend customer service.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class OllamaProvider(LLMProvider):

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate",
        )

    def generate(self, question: str, context: str) -> str:
        language = "French" if is_french(question) else "English"
        prompt = (
            "Answer only from the Boréal Marché context. "
            f"Respond in {language}. "
            "Do not invent information. "
            "If evidence is insufficient, say so. "
            "Keep the answer concise.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException:
            return (
                "Ollama provider is not available. "
                "Start Ollama and verify its connection."
            )


def get_provider(provider_name: str) -> LLMProvider:
    provider_name = provider_name.lower()

    if provider_name == "rule":
        return RuleProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "ollama":
        return OllamaProvider()

    raise ValueError(f"Unknown provider: {provider_name}")


def main():
    provider = get_provider("rule")
    context = (
        "Customers may return eligible products within "
        "30 days of delivery. The product must be unused, "
        "in its original packaging, and accompanied "
        "by proof of purchase."
    )

    for question in [
        "What is your return policy?",
        "Quelle est votre politique de retour ?",
    ]:
        print(question)
        print(provider.generate(question, context))
        print()


if __name__ == "__main__":
    main()
