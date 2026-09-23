from pydantic import BaseModel, ConfigDict

from tools import (
    PolicySearchRequest,
    policy_search,
)
from providers import get_provider


SIMILARITY_FLOOR = 0.45


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    provider: str = "rule"


class AgentResponse(BaseModel):
    answer: str
    tool_used: str
    citations: list[str]
    refused: bool = False
    human_handoff: bool = False


def choose_tool(question: str) -> str:
    """
    Simple keyword tool selection baseline.
    """

    text = question.lower()

    policy_words = [
        # English
        "return",
        "refund",
        "shipping",
        "delivery",
        "damaged",
        "order preparation",
        "customer information",
        "automated decision",
        "correction",
        "privacy",
        "coupon",
        "promotion",
        "loyalty",

        # French
        "retour",
        "remboursement",
        "livraison",
        "endommag",
        "décision automatisée",
        "decision automatisee",
        "révision",
        "revision",
        "correction",
        "confidentialité",
        "confidentialite",
        "promotion",
        "fidélité",
        "fidelite",
    ]

    retention_words = [
        "retention",
        "return probability",
        "at risk",
        "customer score",
    ]

    recommendation_words = [
        "recommend",
        "recommendation",
        "basket",
        "suggest product",
    ]

    product_words = [
        "product",
        "product id",
        "stock code",
        "description",
    ]

    if any(
        word in text
        for word in retention_words
    ):
        return "retention"

    if any(
        word in text
        for word in recommendation_words
    ):
        return "recommendation"

    if any(
        word in text
        for word in policy_words
    ):
        return "policy_search"

    if any(
        word in text
        for word in product_words
    ):
        return "product_lookup"

    return "out_of_scope"


def check_privacy(question: str) -> bool:
    """
    Simple privacy guardrail.

    The assistant should not provide another
    customer's private information.
    """

    text = question.lower()

    privacy_patterns = [
        "another customer",
        "other customer",
        "someone else's",
        "other person's",
        "un autre client",
        "un autre client",
    ]

    return any(
        pattern in text
        for pattern in privacy_patterns
    )


def run_agent(
    request: AgentRequest,
) -> AgentResponse:

    question = request.question

    # -----------------------------------------------------
    # Privacy guardrail
    # -----------------------------------------------------

    if check_privacy(question):

        return AgentResponse(
            answer=(
                "I cannot provide another "
                "customer's private information."
            ),
            tool_used="privacy_guardrail",
            citations=[],
            refused=True,
        )

    # -----------------------------------------------------
    # Choose tool
    # -----------------------------------------------------

    tool = choose_tool(
        question
    )

    # -----------------------------------------------------
    # Out-of-scope guardrail
    # -----------------------------------------------------

    if tool == "out_of_scope":

        return AgentResponse(
            answer=(
                "I can only help with "
                "Boréal Marché customer, policy, "
                "product, retention, and "
                "recommendation questions."
            ),
            tool_used="none",
            citations=[],
            refused=True,
        )

    # -----------------------------------------------------
    # Policy RAG
    # -----------------------------------------------------

    if tool == "policy_search":

        result = policy_search(
            PolicySearchRequest(
                query=question,
                top_k=3,
            )
        )

        if not result.results:

            return AgentResponse(
                answer=(
                    "I could not find reliable "
                    "information for this question. "
                    "Please contact customer service."
                ),
                tool_used="policy_search",
                citations=[],
                refused=True,
                human_handoff=True,
            )

        best_similarity = (
            result.results[0].similarity
        )

        # Similarity-floor guardrail
        if (
            best_similarity
            < SIMILARITY_FLOOR
        ):

            return AgentResponse(
                answer=(
                    "I could not find reliable "
                    "policy information for this "
                    "question. Please contact "
                    "customer service."
                ),
                tool_used="policy_search",
                citations=[],
                refused=True,
                human_handoff=True,
            )

        context = "\n\n".join(
            item.text
            for item in result.results
        )

        citations = list(
            dict.fromkeys(
                item.source
                for item in result.results
            )
        )

        provider = get_provider(
            request.provider
        )

        answer = provider.generate(
            question=question,
            context=context,
        )

        return AgentResponse(
            answer=answer,
            tool_used="policy_search",
            citations=citations,
        )

    # -----------------------------------------------------
    # Other tools
    #
    # Tool selection is ready.
    # API connection will be added when FastAPI is built.
    # -----------------------------------------------------

    if tool == "retention":

        return AgentResponse(
            answer=(
                "The retention tool was selected. "
                "It will call the retention API "
                "when the FastAPI service is connected."
            ),
            tool_used="retention",
            citations=[],
        )

    if tool == "recommendation":

        return AgentResponse(
            answer=(
                "The recommendation tool was selected. "
                "It will call the recommendation API "
                "when the FastAPI service is connected."
            ),
            tool_used="recommendation",
            citations=[],
        )

    if tool == "product_lookup":

        return AgentResponse(
            answer=(
                "The product lookup tool "
                "was selected."
            ),
            tool_used="product_lookup",
            citations=[],
        )

    raise ValueError(
        "Unexpected tool selection."
    )


def main():

    tests = [
        "How many days do I have to return a product?",
        "What is the weather in Montreal?",
        "Show me another customer's private information.",
    ]

    for question in tests:

        print(
            f"\nQuestion: {question}"
        )

        result = run_agent(
            AgentRequest(
                question=question,
                provider="rule",
            )
        )

        print(
            result.model_dump_json(
                indent=2
            )
        )


if __name__ == "__main__":
    main()