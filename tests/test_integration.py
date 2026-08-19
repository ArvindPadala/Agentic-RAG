import unittest
from unittest.mock import patch
from agent import build_search_tool


class TestAgentIntegration(unittest.TestCase):
    @patch('chromadb.Collection')
    @patch('google.genai.Client')
    @patch('boto3.client')
    def test_build_search_tool(self, mock_s3, mock_gemini, mock_collection):
        # Setup mock collection to return a fixed result
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "documents": [["This is a test document."]],
            "metadatas": [[{"source_document": "test_doc", "page": 1}]]
        }
        mock_collection.count.return_value = 1

        search_fn, search_tool = build_search_tool(
            mock_collection, mock_gemini, mock_s3, "test_bucket")

        self.assertEqual(
            search_tool.function_declarations[0].name,
            "search_knowledge_base")
        self.assertIn(
            "Search the document",
            search_tool.function_declarations[0].description)

        # Test the function execution
        result = search_fn("test query")
        self.assertIn("This is a test document.", result)


if __name__ == "__main__":
    unittest.main()
