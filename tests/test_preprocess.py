import unittest

from src.preprocess import clean_text


class CleanTextTests(unittest.TestCase):
    def test_removes_html_tags(self):
        self.assertEqual(clean_text("<b>Great movie</b>"), "great movie")

    def test_preserves_contractions_and_numbers(self):
        self.assertEqual(clean_text("I don't like 2 sequels!!!"), "i don't like 2 sequels")

    def test_handles_non_string_input(self):
        self.assertEqual(clean_text(None), "none")


if __name__ == "__main__":
    unittest.main()
