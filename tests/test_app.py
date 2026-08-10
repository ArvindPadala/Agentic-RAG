import unittest
from app import extract_image_urls, clean_response

class TestAppHelpers(unittest.TestCase):

    def test_extract_image_urls_bare(self):
        text = "Here is an image: https://bucket.amazonaws.com/image.png?sig=123"
        urls = extract_image_urls(text)
        self.assertEqual(urls, ["https://bucket.amazonaws.com/image.png?sig=123"])

    def test_extract_image_urls_markdown(self):
        text = "Here is an image: [Visual Reference](https://bucket.amazonaws.com/image.png?sig=123)"
        urls = extract_image_urls(text)
        self.assertEqual(urls, ["https://bucket.amazonaws.com/image.png?sig=123"])

    def test_clean_response_markdown(self):
        text = "Evidence: [Visual Reference](https://bucket.amazonaws.com/img.png) is here."
        cleaned = clean_response(text)
        self.assertIn("📎 *(see image panel →)*", cleaned)
        self.assertNotIn("https://bucket.amazonaws.com/img.png", cleaned)

    def test_clean_response_bare(self):
        text = "Evidence: https://bucket.amazonaws.com/img.png is here."
        cleaned = clean_response(text)
        self.assertIn("📎 *(see image panel →)*", cleaned)
        self.assertNotIn("https://bucket.amazonaws.com/img.png", cleaned)

if __name__ == '__main__':
    unittest.main()
