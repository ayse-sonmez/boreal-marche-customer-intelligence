import os
from abc import ABC, abstractmethod

import requests

# ---------------------------------------------------------
# Common provider interface
# ---------------------------------------------------------

class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        pass


# ---------------------------------------------------------
# 1. Rule / keyword baseline
# ---------------------------------------------------------

class RuleProvider(LLMProvider):

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        question_lower = question.lower()

        if (
            "return" in question_lower
            or "retour" in question_lower
        ):
            return (
                "Please use the retrieved return "
                "policy information:\n\n"
                + context
            )

        if (
            "shipping" in question_lower
            or "delivery" in question_lower
            or "livraison" in question_lower
        ):
            return (
                "Please use the retrieved shipping "
                "information:\n\n"
                + context
            )

        return (
            "I can only answer using the "
            "available Boréal Marché information."
        )


# ---------------------------------------------------------
# 2. Hosted OpenAI provider
# ---------------------------------------------------------

class OpenAIProvider(LLMProvider):

    def __init__(self):

        self.api_key = os.getenv(
            "OPENAI_API_KEY"
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        if not self.api_key:
            return (
                "OpenAI provider is not configured. "
                "OPENAI_API_KEY is missing."
            )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        prompt = (
            "Answer only from the provided "
            "Boréal Marché context. "
            "If the answer is not in the context, "
            "say that you do not have enough "
            "information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
        )

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0,
            },
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]


# ---------------------------------------------------------
# 3. Local Ollama provider
# ---------------------------------------------------------

class OllamaProvider(LLMProvider):

    def __init__(
        self,
        model: str = "llama3.2",
    ):

        self.model = model

        self.url = (
            "http://localhost:11434/api/generate"
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:

        prompt = (
            "Answer only from the provided "
            "Boréal Marché context. "
            "If the answer is not available, "
            "say that you do not have enough "
            "information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}"
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

            return response.json()[
                "response"
            ]

        except requests.RequestException:

            return (
                "Ollama provider is not available. "
                "Start Ollama locally before using "
                "this provider."
            )


# ---------------------------------------------------------
# Provider selector
# ---------------------------------------------------------

def get_provider(
    provider_name: str,
) -> LLMProvider:

    provider_name = (
        provider_name.lower()
    )

    if provider_name == "rule":
        return RuleProvider()

    if provider_name == "openai":
        return OpenAIProvider()

    if provider_name == "ollama":
        return OllamaProvider()

    raise ValueError(
        f"Unknown provider: {provider_name}"
    )


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

def main():

    print("Testing provider interface...\n")

    provider = get_provider(
        "rule"
    )

    answer = provider.generate(
        question=(
            "How many days do I have "
            "to return a product?"
        ),
        context=(
            "Customers may return eligible "
            "products within 30 days of delivery."
        ),
    )

    print("Provider: rule")
    print(answer)

    print(
        "\nOpenAI and Ollama use the same "
        "generate(question, context) interface."
    )


if __name__ == "__main__":
    main()