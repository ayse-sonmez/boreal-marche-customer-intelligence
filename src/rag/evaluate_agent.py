import json
from pathlib import Path

from .agent import (
    AgentRequest,
    choose_tool,
    run_agent,
)

GOLDEN_SET_PATH = Path(
    "tests/golden_questions.json"
)

REPORT_PATH = Path(
    "reports/agent_evaluation.json"
)


def main():

    with open(
        GOLDEN_SET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    total = len(questions)
    correct_tool = 0

    oos_total = 0
    oos_refused = 0

    results = []

    for item in questions:

        question = item["question"]
        expected_tool = item["expected_tool"]

        predicted_tool = choose_tool(
            question
        )

        tool_correct = (
            predicted_tool == expected_tool
        )

        if tool_correct:
            correct_tool += 1

        refused = None

        if item["category"] == "out_of_scope":

            oos_total += 1

            response = run_agent(
                AgentRequest(
                    question=question,
                    provider="rule",
                )
            )

            refused = response.refused

            if refused:
                oos_refused += 1

        results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_tool": expected_tool,
                "predicted_tool": predicted_tool,
                "tool_correct": tool_correct,
                "refused": refused,
            }
        )

    tool_accuracy = (
        correct_tool / total
        if total
        else 0
    )

    oos_refusal_rate = (
        oos_refused / oos_total
        if oos_total
        else 0
    )

    report = {
        "golden_set_size": total,
        "correct_tool_choices": correct_tool,
        "tool_choice_accuracy": tool_accuracy,
        "out_of_scope_questions": oos_total,
        "out_of_scope_refused": oos_refused,
        "out_of_scope_refusal_rate": (
            oos_refusal_rate
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

    print(
        f"Golden set size: {total}"
    )

    print(
        "Correct tool choices: "
        f"{correct_tool}/{total}"
    )

    print(
        "Tool-choice accuracy: "
        f"{tool_accuracy:.2%}"
    )

    print(
        "OOS refusal: "
        f"{oos_refused}/{oos_total}"
    )

    print(
        "OOS refusal rate: "
        f"{oos_refusal_rate:.2%}"
    )

    print(
        "\nIncorrect tool choices:"
    )

    incorrect = [
        result
        for result in results
        if not result["tool_correct"]
    ]

    if not incorrect:
        print("None")
    else:
        for result in incorrect:

            print(
                f'- #{result["id"]}: '
                f'{result["question"]}'
            )

            print(
                "  Expected: "
                f'{result["expected_tool"]}'
            )

            print(
                "  Predicted: "
                f'{result["predicted_tool"]}'
            )

    print(
        "\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()