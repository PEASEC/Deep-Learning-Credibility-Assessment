import unittest
from .BertPreprocessor import preprocess_bert

str1 = "Hello #World, today is a nice day!"
str2 = "HELLO WORLD, today is a nice day!"
str3 = "Hello @World, today is a really nice day!"
str4 = "Hello World, have a look at https://xkcd.com - it's awesome!"
str5 = "@World @Universe @God Today is a #beautiful day with #awesome COMICS at https://xkcd.com"


class TestSum(unittest.TestCase):
    def test_url(self):
        def fn(text: str):
            return preprocess_bert(
                text,
                url_token="[url]",
                mention_token=None,
                hashtag_token=None,
                allcaps_token=None,
                replace_emojis=False,
                replace_smileys=False
            )

        self.assertEqual(str1, fn(str1))
        self.assertEqual(str2, fn(str2))
        self.assertEqual(str3, fn(str3))
        self.assertEqual("Hello World, have a look at [url] - it's awesome!", fn(str4))
        self.assertEqual(
            "@World @Universe @God Today is a #beautiful day with #awesome COMICS at [url]",
            fn(str5)
        )

    def test_mention(self):
        def fn(text: str):
            return preprocess_bert(
                text,
                url_token=None,
                mention_token="[user]",
                hashtag_token=None,
                allcaps_token=None,
                replace_emojis=False,
                replace_smileys=False
            )

        self.assertEqual(str1, fn(str1))
        self.assertEqual(str2, fn(str2))
        self.assertEqual("Hello [user], today is a really nice day!", fn(str3))
        self.assertEqual(str4, fn(str4))
        self.assertEqual(
            "[user] [user] [user] Today is a #beautiful day with #awesome COMICS at https://xkcd.com",
            fn(str5)
        )

    def test_caps(self):
        def fn(text: str):
            return preprocess_bert(
                text,
                url_token=None,
                mention_token=None,
                hashtag_token=None,
                allcaps_token="[caps]",
                replace_emojis=False,
                replace_smileys=False
            )

        self.assertEqual(str1, fn(str1))
        self.assertEqual("[caps] hello [caps] world, today is a nice day!", fn(str2))
        self.assertEqual(str3, fn(str3))
        self.assertEqual(str4, fn(str4))
        self.assertEqual(
            "@World @Universe @God Today is a #beautiful day with #awesome [caps] comics at https://xkcd.com",
            fn(str5)
        )

    def test_hashtag(self):
        def fn(text: str):
            return preprocess_bert(
                text,
                url_token=None,
                mention_token=None,
                hashtag_token="[hashtag]",
                allcaps_token=None,
                replace_emojis=False,
                replace_smileys=False
            )

        self.assertEqual("Hello [hashtag], today is a nice day!", fn(str1))
        self.assertEqual(str2, fn(str2))
        self.assertEqual(str3, fn(str3))
        self.assertEqual(str4, fn(str4))
        self.assertEqual(
            "@World @Universe @God Today is a [hashtag] day with [hashtag] COMICS at https://xkcd.com",
            fn(str5)
        )

    def test_all(self):
        def fn(text: str):
            return preprocess_bert(
                text,
                url_token="[url]",
                mention_token="[user]",
                hashtag_token="[hashtag]",
                allcaps_token="[caps]",
                replace_emojis=False,
                replace_smileys=False
            )

        self.assertEqual(
            "[user] [user] [user] Today is a [hashtag] day with [hashtag] [caps] comics at [url]",
            fn(str5)
        )


if __name__ == '__main__':
    unittest.main()
