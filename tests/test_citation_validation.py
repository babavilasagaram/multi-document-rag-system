
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from app.generation import generator


def test_citation_validation_and_formatting():
    fake_chunks = [
        {
            "text": "Transformers use attention mechanisms.",
            "metadata": {
                "source": "test.pdf",
                "page": 3
            },
            "similarity": 0.85
        }
    ]

    fake_llm = RunnableLambda(
        lambda inputs: "Transformers use attention [1]."
    )

    with patch("app.generation.generator.llm", fake_llm):
        result = generator.generate_answer(
            "What do Transformers use?",
            fake_chunks
        )

    assert "[Source: test.pdf, Page: 3]" in result
    assert "did not contain citations" not in result


def test_invalid_citation_is_rejected():
    sources = {
        "1": {
            "source": "test.pdf",
            "page": 3
        }
    }

    result = generator.validate_and_format_citations(
        "Transformers use attention [2].",
        sources
    )

    assert "invalid citations" in result


def test_missing_citation_is_rejected():
    sources = {
        "1": {
            "source": "test.pdf",
            "page": 3
        }
    }

    result = generator.validate_and_format_citations(
        "Transformers use attention.",
        sources
    )

    assert "did not contain citations" in result
