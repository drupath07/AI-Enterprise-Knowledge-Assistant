from backend.rag_chat import ask_question


# ============================================================
# RAG EVALUATION TESTS
# ============================================================

tests = [

    {
        "name": "Annual Leave",
        "question": "How many days of annual leave do employees get?",
        "expected": "24 days",
        "source": "company_policy.txt"
    },

    {
        "name": "Work From Home",
        "question": "How many days of work from home are allowed?",
        "expected": "2 days",
        "source": "company_policy.txt"
    },

    {
        "name": "Emergency Leave",
        "question": "What should employees do for emergency leave?",
        "expected": "manager",
        "source": "company_policy.txt"
    },

    {
        "name": "Employee Benefits",
        "question": "What employee benefits are available?",
        "expected": "health insurance",
        "source": "benefits_policy.txt"
    },

    {
        "name": "Performance Bonus",
        "question": "Are employees eligible for a performance bonus?",
        "expected": "performance bonus",
        "source": "benefits_policy.txt"
    },

    {
        "name": "HR Policy",
        "question": "How often are performance reviews conducted?",
        "expected": "twice a year",
        "source": "hr_policy.txt"
    },

    {
        "name": "Security Incident",
        "question": "What should employees do if they suspect a security incident?",
        "expected": "IT",
        "source": "it_security_policy.txt"
    },

    {
        "name": "Unknown Information",
        "question": "What is the employee salary?",
        "expected": "I don't have enough information",
        "source": None
    }

]


# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 70)
print("AI ENTERPRISE KNOWLEDGE ASSISTANT")
print("RAG EVALUATION")
print("=" * 70)
print()


passed = 0
failed = 0


for number, test in enumerate(tests, start=1):

    print("-" * 70)

    print(
        f"Test {number}: {test['name']}"
    )

    print(
        f"Question: {test['question']}"
    )

    print()


    # --------------------------------------------------------
    # ASK RAG SYSTEM
    # --------------------------------------------------------

    result = ask_question(
        test["question"]
    )


    answer = result["answer"]

    sources = result["sources"]


    print(
        f"Answer: {answer}"
    )

    print(
        f"Sources: {sources}"
    )


    # --------------------------------------------------------
    # CHECK ANSWER
    # --------------------------------------------------------

    answer_correct = (
        test["expected"].lower()
        in answer.lower()
    )


    # --------------------------------------------------------
    # CHECK SOURCE
    # --------------------------------------------------------

    if test["source"] is None:

        source_correct = (
            len(sources) == 0
            or test["source"] is None
        )

    else:

        source_correct = (
            test["source"] in sources
        )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if answer_correct and source_correct:

        print()
        print("RESULT: PASS")

        passed += 1

    else:

        print()
        print("RESULT: FAIL")

        if not answer_correct:

            print(
                "Reason: Expected information was not found "
                "in the answer."
            )


        if not source_correct:

            print(
                "Reason: Expected source was not found."
            )

        failed += 1


    print()


# ============================================================
# FINAL SUMMARY
# ============================================================

total = len(tests)

accuracy = (
    passed / total
) * 100


print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Total Tests : {total}"
)

print(
    f"Passed      : {passed}"
)

print(
    f"Failed      : {failed}"
)

print(
    f"Accuracy    : {accuracy:.2f}%"
)

print("=" * 70)