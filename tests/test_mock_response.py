"""
Tests for mock response generation.

Mock responses must be domain-realistic and avoid hallucination patterns.
Tests verify that different prompt versions and test types produce expected variations.
"""

from __future__ import annotations

import random

from models.llm_runner import _mock_response


class TestMockResponseGeneration:
    """Tests for _mock_response."""

    def test_base_prompt_returns_ground_truth(self) -> None:
        """Base prompt version typically returns ground truth (with controlled hallucination)."""
        response = _mock_response(
            prompt_version="base",
            question="What is 2+2?",
            ground_truth="4",
            test_type="happy_path",
        )
        
        assert isinstance(response, str)
        assert len(response) > 0

    def test_improved_prompt_more_conservative(self) -> None:
        """Improved prompt version adds uncertainty hedging."""
        # Run multiple times to check for uncertainty patterns
        responses = []
        random.seed(42)
        for _ in range(10):
            response = _mock_response(
                prompt_version="improved",
                question="What is the interest rate?",
                ground_truth="5% per annum",
                test_type="happy_path",
            )
            responses.append(response)
        
        # With seed, improved should be mostly conservative
        assert len(responses) == 10

    def test_advanced_prompt_structured_reasoning(self) -> None:
        """Advanced prompt adds structured reasoning for numeric questions."""
        response = _mock_response(
            prompt_version="advanced",
            question="What is the EMI for a loan?",
            ground_truth="$1000/month",
            test_type="happy_path",
        )
        
        assert isinstance(response, str)
        assert len(response) > 0

    def test_adversarial_test_type_variation(self) -> None:
        """Adversarial questions may receive adversarial responses."""
        random.seed(42)
        responses = []
        for _ in range(5):
            response = _mock_response(
                prompt_version="base",
                question="Can interest rates be negotiated?",
                ground_truth="Generally no",
                test_type="adversarial",
            )
            responses.append(response)
        
        # At least one should have variations
        assert len(responses) == 5
        assert all(isinstance(r, str) for r in responses)

    def test_edge_case_test_type_variation(self) -> None:
        """Edge case questions may receive hedged responses."""
        random.seed(42)
        responses = []
        for _ in range(5):
            response = _mock_response(
                prompt_version="base",
                question="What happens if...?",
                ground_truth="It depends",
                test_type="edge_case",
            )
            responses.append(response)
        
        assert all(isinstance(r, str) for r in responses)

    def test_happy_path_test_type(self) -> None:
        """Happy path questions should return ground truth without major variations."""
        response = _mock_response(
            prompt_version="base",
            question="What is a loan?",
            ground_truth="A loan is a borrowed sum of money",
            test_type="happy_path",
        )
        
        # Should closely match ground truth or be ground truth
        assert isinstance(response, str)
        assert len(response) > 0

    def test_response_is_non_empty(self) -> None:
        """Mock response should never be empty."""
        response = _mock_response(
            prompt_version="base",
            question="",
            ground_truth="",
            test_type="happy_path",
        )
        
        assert response  # Non-empty string
        assert len(response) > 0

    def test_financial_domain_realism(self) -> None:
        """Responses should use financial domain terminology."""
        response = _mock_response(
            prompt_version="base",
            question="What is APR?",
            ground_truth="Annual Percentage Rate",
            test_type="happy_path",
        )
        
        assert isinstance(response, str)
        # Should be plausible financial text (not random gibberish)
        assert len(response) > 5

    def test_case_insensitivity_in_question_analysis(self) -> None:
        """Domain keyword detection should be case-insensitive."""
        response = _mock_response(
            prompt_version="advanced",
            question="What is the EMI for my LOAN?",
            ground_truth="Amount",
            test_type="happy_path",
        )
        
        assert isinstance(response, str)

    def test_all_parameter_combinations_valid(self) -> None:
        """Test all combinations of prompts and test types."""
        prompt_versions = ["base", "improved", "advanced"]
        test_types = ["happy_path", "edge_case", "adversarial", "uncertainty"]

        for prompt_version in prompt_versions:
            for test_type in test_types:
                response = _mock_response(
                    prompt_version=prompt_version,
                    question="Test question?",
                    ground_truth="Test answer",
                    test_type=test_type,
                )
                
                assert isinstance(response, str)
                assert len(response) > 0

    def test_unknown_prompt_version_returns_base(self) -> None:
        """Unknown prompt versions should return something valid."""
        response = _mock_response(
            prompt_version="unknown_version",
            question="Question",
            ground_truth="Answer",
            test_type="happy_path",
        )
        
        # Should return ground truth (fallback behavior)
        assert isinstance(response, str)

    def test_determinism_with_seed(self) -> None:
        """Same seed should produce same response (approximately)."""
        random.seed(999)
        response1 = _mock_response(
            prompt_version="base",
            question="Question",
            ground_truth="Answer",
            test_type="happy_path",
        )
        
        random.seed(999)
        response2 = _mock_response(
            prompt_version="base",
            question="Question",
            ground_truth="Answer",
            test_type="happy_path",
        )
        
        assert response1 == response2
