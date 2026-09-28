import json
import time
from pathlib import Path

from agent import AgentRequest, run_agent

GOLDEN_SET_PATH = Path(
    "tests/golden_questions.json"
)

REPORT_PATH = Path(
    "reports/rag_evaluation.json"
)


def main():

    with open(
        GOLDEN_SET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    # Policy questions are the relevant
    # questions for RAG grounding evaluation.
    policy_questions = [
        item
        for item in questions
        if item["expected_tool"]
        == "policy_search"
    ]

    results = []

    correct_source = 0
    total_latency = 0.0

    for item in policy_questions:

        start = time.perf_counter()

        response = run_agent(
            AgentRequest(
                question=item["question"],
                provider="rule",
            )
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        total_latency += latency_ms

        expected_source = item.get(
            "expected_source"
        )

        source_found = (
            expected_source
            in response.citations
        )

        if source_found:
            correct_source += 1

        results.append(
            {
                "id": item["id"],
                "language": item["language"],
                "question": item["question"],
                "expected_source": (
                    expected_source
                ),
                "citations": (
                    response.citations
                ),
                "source_grounded": (
                    source_found
                ),
                "latency_ms": round(
                    latency_ms,
                    2,
                ),
                "provider": "rule",
                "cost_usd": 0.0,
            }
        )

        print(
            f'#{item["id"]} '
            f'grounded={source_found} '
            f'latency={latency_ms:.2f} ms'
        )

    total = len(policy_questions)

    groundedness = (
        correct_source / total
        if total
        else 0
    )

    average_latency = (
        total_latency / total
        if total
        else 0
    )

    report = {
        "provider": "rule",
        "questions_evaluated": total,
        "correct_source_citations": (
            correct_source
        ),
        "source_groundedness": (
            groundedness
        ),
        "average_latency_ms": round(
            average_latency,
            2,
        ),
        "cost_per_query_usd": 0.0,
        "cost_note": (
            "The rule baseline does not call "
            "a paid language model, so its "
            "provider cost is $0 per query."
        ),
        "groundedness_note": (
            "For this baseline evaluation, "
            "groundedness is measured by "
            "whether the expected source "
            "document appears in the citations."
        ),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n--- RAG Evaluation ---")

    print(
        "Questions evaluated: "
        f"{total}"
    )

    print(
        "Correct source citations: "
        f"{correct_source}/{total}"
    )

    print(
        "Source groundedness: "
        f"{groundedness:.2%}"
    )

    print(
        "Average latency: "
        f"{average_latency:.2f} ms"
    )

    print(
        "Rule provider cost/query: "
        "$0.00"
    )

    print(
        "\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()