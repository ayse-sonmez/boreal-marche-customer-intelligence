import re

from pydantic import BaseModel, ConfigDict

from .tools import (
    PolicySearchRequest,
    RetentionRequest,
    RecommendationRequest,
    ProductLookupRequest,
    policy_search,
    retention_score,
    recommendations,
    product_lookup,
)
from .providers import get_provider


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
    """

    text = question.lower()

    privacy_patterns = [
        "another customer",
        "other customer",
        "someone else's",
        "other person's",
        "un autre client",
    ]

    return any(
        pattern in text
        for pattern in privacy_patterns
    )


def extract_numbers(
    question: str,
) -> list[str]:
    """
    Extract numeric IDs from the question.
    """

    return re.findall(
        r"\b\d+\b",
        question,
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

    tool = choose_tool(question)

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
    # Retention tool
    # -----------------------------------------------------

    if tool == "retention":

        numbers = extract_numbers(
            question
        )

        if not numbers:

            return AgentResponse(
                answer=(
                    "Please provide a customer ID "
                    "for the retention prediction."
                ),
                tool_used="retention",
                citations=[],
            )

        customer_id = int(
            numbers[0]
        )

        result = retention_score(
            RetentionRequest(
                customer_id=customer_id
            )
        )

        risk_text = (
            "at risk"
            if result.at_risk
            else "not at risk"
        )

        answer = (
            f"Customer {result.customer_id} "
            f"has a return probability of "
            f"{result.return_probability:.1%}. "
            f"The decision threshold is "
            f"{result.threshold:.0%}, so the "
            f"customer is {risk_text}."
        )

        return AgentResponse(
            answer=answer,
            tool_used="retention",
            citations=[],
        )

    # -----------------------------------------------------
    # Recommendation tool
    # -----------------------------------------------------

    if tool == "recommendation":

        numbers = extract_numbers(
            question
        )

        if not numbers:

            return AgentResponse(
                answer=(
                    "Please provide at least one "
                    "product ID for the basket."
                ),
                tool_used="recommendation",
                citations=[],
            )

        result = recommendations(
            RecommendationRequest(
                basket=numbers,
                top_k=5,
            )
        )

        if not result.recommendations:

            return AgentResponse(
                answer=(
                    "I could not find a matching "
                    "recommendation for this basket."
                ),
                tool_used="recommendation",
                citations=[],
            )

        lines = []

        for item in result.recommendations:

            lines.append(
                f"{item.product} because "
                f"the basket contains "
                f"{item.because_product}"
            )

        answer = (
            "Recommended product(s): "
            + "; ".join(lines)
            + "."
        )

        return AgentResponse(
            answer=answer,
            tool_used="recommendation",
            citations=[],
        )

    # -----------------------------------------------------
    # Product lookup tool
    # -----------------------------------------------------

    if tool == "product_lookup":

        numbers = extract_numbers(
            question
        )

        if not numbers:

            return AgentResponse(
                answer=(
                    "Please provide a product ID "
                    "or stock code."
                ),
                tool_used="product_lookup",
                citations=[],
            )

        product_id = numbers[0]

        result = product_lookup(
            ProductLookupRequest(
                product_id=product_id
            )
        )

        if not result.found:

            return AgentResponse(
                answer=(
                    f"Product {product_id} "
                    "was not found."
                ),
                tool_used="product_lookup",
                citations=[],
            )

        return AgentResponse(
            answer=(
                f"Product {result.product_id}: "
                f"{result.description}."
            ),
            tool_used="product_lookup",
            citations=[],
        )

    raise ValueError(
        "Unexpected tool selection."
    )


def main():

    tests = [
        (
            "How many days do I have "
            "to return a product?"
        ),
        "Is customer 12346 at risk?",
        (
            "Recommend a product for "
            "basket 22748."
        ),
        (
            "What is the description "
            "of product 22745?"
        ),
        "What is the weather in Montreal?",
        (
            "Show me another customer's "
            "private information."
        ),
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