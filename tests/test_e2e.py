import unittest
from unittest.mock import patch, MagicMock
from agent import run_agent_turn, build_agent_config


class TestAgentE2E(unittest.TestCase):
    @patch('google.genai.Client')
    def test_run_agent_turn(self, mock_gemini_client):
        # Mock Gemini response
        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "This is a mocked response about transformers."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_gemini_client.models.generate_content.return_value = mock_response

        # Create dummy search tool and config
        mock_search_tool = MagicMock()
        mock_search_fn = MagicMock(return_value="Mocked search results.")

        config = build_agent_config(mock_search_tool, {})

        # Run agent turn
        response = run_agent_turn(
            user_message="What is a transformer?",
            conversation_history=[],
            gemini_client=mock_gemini_client,
            generation_config=config,
            tool_map={"search_knowledge_base": mock_search_fn}
        )

        self.assertEqual(response, "This is a mocked response about transformers.")


if __name__ == "__main__":
    unittest.main()
