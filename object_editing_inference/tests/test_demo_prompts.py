import unittest

from fastapi import HTTPException

from app.main import parse_prompts


class DemoPromptTests(unittest.TestCase):
    def test_table_prompt_adds_furniture_synonyms(self):
        self.assertEqual(
            parse_prompts("table"),
            ["table", "coffee table", "dining table", "side table", "desk"],
        )

    def test_sofa_and_couch_aliases_are_deduplicated(self):
        self.assertEqual(parse_prompts("sofa, couch"), ["sofa", "couch"])

    def test_user_input_is_still_limited_to_five_prompts(self):
        with self.assertRaises(HTTPException):
            parse_prompts("one,two,three,four,five,six")


if __name__ == "__main__":
    unittest.main()
