import unittest
from app import extract_image_urls, clean_response


class TestAppUnit(unittest.TestCase):
    def test_extract_image_urls(self):
        text = "Here is a picture: [View Reference](https://arvind-agentic-rag-2026.s3.amazonaws.com/image.png?AWSAccessKeyId=123) and some text."
        urls = extract_image_urls(text)
        self.assertEqual(len(urls), 1)
        self.assertTrue(urls[0].startswith("https://arvind-agentic-rag-2026.s3.amazonaws.com/image.png"))

    def test_clean_response(self):
        text = "Here is a picture: [View Reference](https://arvind-agentic-rag-2026.s3.amazonaws.com/image.png?AWSAccessKeyId=123) and some text."
        cleaned = clean_response(text)
        self.assertNotIn("https://arvind-agentic-rag-2026.s3.amazonaws.com", cleaned)
        self.assertIn("Here is a picture:", cleaned)


if __name__ == "__main__":
    unittest.main()
