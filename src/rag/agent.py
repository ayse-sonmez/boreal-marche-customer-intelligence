import re
import unicodedata

from pydantic import BaseModel, ConfigDict

from .providers import get_provider, is_french
from .tools import (
    PolicySearchRequest,
    ProductLookupRequest,
    RecommendationRequest,
    RetentionRequest,
    policy_search,
    product_lookup,
    recommendations,
    retention_score,
)

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


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def choose_tool(question: str) -> str:
    text = normalize_text(question)
    retention_words = ["retention", "return probability", "at risk", "customer score", "fidelisation", "probabilite de retour", "client a risque", "score client"]
    recommendation_words = ["recommend", "basket", "suggest product", "recommand", "panier", "suggerer un produit"]
    policy_words = ["return", "refund", "shipping", "delivery", "damaged", "order preparation", "customer information", "automated decision", "correction", "privacy", "coupon", "promotion", "loyalty", "retour", "remboursement", "livraison", "endommag", "decision automatisee", "revision", "confidentialite", "fidelite", "politique", "echang"]
    product_words = ["product", "product id", "stock code", "description", "produit", "code article"]

    if any(word in text for word in retention_words):
        return "retention"
    if any(word in text for word in recommendation_words):
        return "recommendation"
    if any(word in text for word in policy_words):
        return "policy_search"
    if any(word in text for word in product_words):
        return "product_lookup"
    return "out_of_scope"


def check_privacy(question: str) -> bool:
    text = normalize_text(question)
    patterns = ["another customer", "other customer", "someone else's", "other person's", "un autre client", "d'un autre client", "un autre utilisateur"]
    return any(pattern in text for pattern in patterns)


def extract_numbers(question: str) -> list[str]:
    return re.findall(r"\b\d+\b", question)


def localized(question: str, english: str, french: str) -> str:
    return french if is_french(question) else english


def policy_queries(question: str) -> list[str]:
    """Expand only known return-policy intents; retain the original query."""
    queries = [question]
    normalized = normalize_text(question)
    if "retour" in normalized or "return" in normalized:
        queries.append(
            "Combien de jours ai-je pour retourner un produit?"
            if is_french(question)
            else "How many days do I have to return a product?"
        )
    return queries


def search_reliable_policy(question: str):
    """Keep only chunks that meet the existing similarity floor.

    Select the strongest successful query, avoiding mixed contexts from
    unrelated searches. No similarity scores are artificially boosted.
    """
    best = []
    best_score = float("-inf")
    for query in policy_queries(question):
        response = policy_search(PolicySearchRequest(query=query, top_k=3))
        reliable = [item for item in response.results if item.similarity >= SIMILARITY_FLOOR]
        if reliable and reliable[0].similarity > best_score:
            best = reliable
            best_score = reliable[0].similarity
    return best


def run_agent(request: AgentRequest) -> AgentResponse:
    question = request.question.strip()
    if not question:
        return AgentResponse(
            answer=localized(question, "Please enter a question.", "Veuillez saisir une question."),
            tool_used="none", citations=[], refused=True,
        )

    if check_privacy(question):
        return AgentResponse(
            answer=localized(
                question,
                "I cannot provide another customer's private information.",
                "Je ne peux pas divulguer les renseignements personnels d'un autre client.",
            ),
            tool_used="privacy_guardrail", citations=[], refused=True,
        )

    tool = choose_tool(question)
    if tool == "out_of_scope":
        return AgentResponse(
            answer=localized(
                question,
                "I can help with Boréal Marché policies, products, retention and recommendations.",
                "Je peux vous aider avec les politiques, les produits, la fidélisation et les recommandations de Boréal Marché.",
            ),
            tool_used="none", citations=[], refused=True,
        )

    if tool == "policy_search":
        results = search_reliable_policy(question)
        if not results:
            return AgentResponse(
                answer=localized(
                    question,
                    "I could not find reliable policy information for this question. Please contact customer service.",
                    "Je n'ai pas trouvé d'information fiable pour cette question. Veuillez contacter le service à la clientèle.",
                ),
                tool_used="policy_search", citations=[], refused=True, human_handoff=True,
            )
        context = "\n\n".join(item.text for item in results)
        citations = list(dict.fromkeys(item.source for item in results))
        provider = get_provider(request.provider)
        answer = provider.generate(question=question, context=context)
        return AgentResponse(answer=answer, tool_used="policy_search", citations=citations)

    if tool == "retention":
        numbers = extract_numbers(question)
        if not numbers:
            return AgentResponse(
                answer=localized(question, "Please provide a customer ID for the retention prediction.", "Veuillez fournir un identifiant client pour la prédiction de fidélisation."),
                tool_used="retention", citations=[],
            )
        result = retention_score(RetentionRequest(customer_id=int(numbers[0])))
        if is_french(question):
            risk = "à risque" if result.at_risk else "non à risque"
            answer = f"Le client {result.customer_id} a une probabilité de retour de {result.return_probability:.1%}. Le seuil de décision est de {result.threshold:.0%} : le client est {risk}."
        else:
            risk = "at risk" if result.at_risk else "not at risk"
            answer = f"Customer {result.customer_id} has a return probability of {result.return_probability:.1%}. The decision threshold is {result.threshold:.0%}, so the customer is {risk}."
        return AgentResponse(answer=answer, tool_used="retention", citations=[])

    if tool == "recommendation":
        numbers = extract_numbers(question)
        if not numbers:
            return AgentResponse(
                answer=localized(question, "Please provide at least one product ID for the basket.", "Veuillez fournir au moins un code produit pour le panier."),
                tool_used="recommendation", citations=[],
            )
        result = recommendations(RecommendationRequest(basket=numbers, top_k=5))
        if not result.recommendations:
            return AgentResponse(
                answer=localized(question, "I could not find a matching recommendation for this basket.", "Je n'ai trouvé aucune recommandation correspondante pour ce panier."),
                tool_used="recommendation", citations=[],
            )
        if is_french(question):
            lines = [f"{item.product} (car le panier contient {item.because_product})" for item in result.recommendations]
            answer = "Produits recommandés : " + "; ".join(lines) + "."
        else:
            lines = [f"{item.product} because the basket contains {item.because_product}" for item in result.recommendations]
            answer = "Recommended product(s): " + "; ".join(lines) + "."
        return AgentResponse(answer=answer, tool_used="recommendation", citations=[])

    if tool == "product_lookup":
        numbers = extract_numbers(question)
        if not numbers:
            return AgentResponse(
                answer=localized(question, "Please provide a product ID or stock code.", "Veuillez fournir un code produit."),
                tool_used="product_lookup", citations=[],
            )
        product_id = numbers[0]
        result = product_lookup(ProductLookupRequest(product_id=product_id))
        if not result.found:
            answer = localized(question, f"Product {product_id} was not found.", f"Le produit {product_id} est introuvable.")
        else:
            answer = localized(question, f"Product {result.product_id}: {result.description}.", f"Produit {result.product_id} : {result.description}.")
        return AgentResponse(answer=answer, tool_used="product_lookup", citations=[])

    raise ValueError("Unexpected tool selection.")


def main():
    questions = [
        "What is your return policy?",
        "Quelle est votre politique de retour ?",
        "How many days do I have to return a product?",
        "What is the weather in Montreal?",
        "Show me another customer's private information.",
    ]
    for question in questions:
        print(f"\nQuestion: {question}")
        result = run_agent(AgentRequest(question=question, provider="rule"))
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
