import unittest
from .GloveTokenizer import tokenize
from .FilePositionCache import FilePositionCache, FilePositionCacheBuilder
from .GloveEmbeddings import load_embedding
import os
import random
import string
from typing import Dict
from tqdm import tqdm


class TestTokenize(unittest.TestCase):
    def test_hashtag(self):
        self.assertEqual(
            ["hello", "<hashtag>", "world", ",", "today", "is", "a", "nice", "day", "!"],
            tokenize("Hello #World, today is a nice day!")
        )

    def test_allcaps(self):
        self.assertEqual(
            ["hello", "<allcaps>", "world", "<allcaps>", ",", "today",
             "is", "a", "nice", "day", "!"],
            tokenize("HELLO WORLD, today is a nice day!")
        )

    def test_mention(self):
        self.assertEqual(
            ["hello", "<user>", ",", "today", "is", "a", "really", "nice", "day", "!"],
            tokenize("Hello @World, today is a really nice day!")
        )

    def test_url(self):
        self.assertEqual(
            ["hello", "world", ",", "have", "a", "look", "at", "<url>", "-", "it", "'", "s", "awesome", "!"],
            tokenize("Hello World, have a look at https://xkcd.com - it's awesome!")
        )

    def test_all(self):
        self.assertEqual(
            ["<user>", "<user>", "<user>", "today", "is", "a", "<hashtag>", "beautiful", "day", "with", "<hashtag>",
             "awesome", "comics", "<allcaps>", "at", "<url>"],
            tokenize("@World @Universe @God Today is a #beautiful day with #awesome COMICS at https://xkcd.com")
        )


class TestCache(unittest.TestCase):
    def check_invariant(self, cache):
        self.assertEqual(
            len(cache.cache), len(cache.ordered_cache),
            "Cache and ordered cache must have the same length"
        )

        # Ordered Cache must be ordered
        for i in range(1, len(cache.ordered_cache)):
            self.assertGreaterEqual(
                cache.ordered_cache[i - 1].info.accesses,
                cache.ordered_cache[i].info.accesses,
                "Failed invariant: ordered_cache must be ordered"
            )

        # Positions are provided in ordered cache
        for i in range(len(cache.ordered_cache)):
            self.assertEqual(
                cache.ordered_cache[i].cache_position,
                i,
                "Failed invariant: All elements in array have position associated"
            )

    def test_auto(self):
        text = "This was a triumph. I am making a note her: HUGE SUCCESS. " + \
               "It is hard to overstate my satisfaction. " + \
               "We do what we must because we can. " + \
               "For the good of all of us. Except the ones who are dead. " + \
               "But there is no sense crying over every mistake. " + \
               "You just keep on trying until you run out of cake. " + \
               "And the Science gets done and you make a neat gone. " + \
               "For the people how are still alive. " + \
               "I am not even angry. I am being so sincere right niw. " + \
               "Even though you broke my heat. And killed me. " + \
               "And tore me to pieces. And throw every piece into a fire. " + \
               "As they burned it hurt because I was so happy for you. " + \
               "Now these points of data make a beautiful line and we are " + \
               "out of beta we are releasing on time."

        word_list = text.replace(".", " .").split(" ")

        import numpy as np
        np.random.seed(42)
        vectors: Dict[str, np.ndarray] = dict((word, np.random.rand(50)) for word in set(word_list))

        capacity = 10
        builder: FilePositionCacheBuilder[str, np.ndarray] = FilePositionCacheBuilder(capacity)
        for index, word in enumerate(vectors.keys()):
            builder.add_item(word, index)

        was_full = False
        cache: FilePositionCache[str, np.ndarray] = builder.build()
        for index, word in enumerate(word_list):
            if cache.has_value(word):
                self.assertEqual(cache.get_value(word).tolist(), vectors[word].tolist(), "Word cached incorrectly")
            else:
                vector = vectors[word]
                if cache.has_pointer(word):
                    cache.get_pointer(word)
                    cache.set_value(word, vector)

            self.check_invariant(cache)

            # Once full, it must be full all the time
            if len(cache.cache) == capacity:
                was_full = True

            if was_full:
                self.assertEqual(len(cache.cache), capacity, "Once full, cache must have capacity size")

        self.assertTrue(was_full, "Cache must be full")

    @staticmethod
    def get_random_string(length) -> str:
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def test_on_large_scale(self):
        random.seed(42)
        word_list = [self.get_random_string(15) for i in range(10_000)]

        builder: FilePositionCacheBuilder[str, int] = FilePositionCacheBuilder(500)
        for word in word_list:
            builder.add_item(word, random.randint(1, 10_000))

        cache = builder.build()

        prefix = []
        prefix.extend(list(range(10)))
        prefix.extend([1] * 5)  # Positions 0-1000 are 5 times more likely
        for i in tqdm(range(10_000), desc="Executing large scale test"):
            index = random.choice(prefix) * 1000 + random.randint(0, 999)
            assert index < 10000

            word = word_list[index]
            if not cache.has_value(word):
                cache.set_value(word, random.randint(0, 10_000))

            self.check_invariant(cache)

        self.assertEqual(len(cache.cache), 500, "Cache not fully used")
        print("Done")

    def test_cache(self):
        builder: FilePositionCacheBuilder[str, str] = FilePositionCacheBuilder(4)
        cache = builder \
            .add_item("we", 0) \
            .add_item("do", 1) \
            .add_item("what", 2) \
            .add_item("we", 3) \
            .add_item("must", 4) \
            .add_item("because", 5) \
            .add_item("we", 6) \
            .add_item("can", 7) \
            .add_item(".", 8) \
            .add_item("there", 9) \
            .add_item("is", 10) \
            .add_item("no", 11) \
            .add_item("sense", 12) \
            .add_item("crying", 13) \
            .add_item("over", 14) \
            .add_item("mistake", 15) \
            .add_item("keep", 16) \
            .add_item("on", 17) \
            .add_item("trying", 18) \
            .build()

        special_meanings = {
            "there": "clara",
            "is": "maria",
            "no": "daniel",
            "sense": "josef",
            "crying": "hendrik",
            "we": "marc",
            "do": "hannah",
            "what": "andreas"
        }

        basis = ["there", "is", "no", "sense", "crying"]
        additions = ["there", "there", "there"]
        additions2 = ["we", "do", "what", "we"]
        for word in basis:
            self.assertEqual(cache.has_value(word), False)
            cache.set_value(word, special_meanings[word])

        self.assertEqual(cache.get_cached_keys(), ["there", "is", "no", "sense"])

        for word in additions:
            self.assertEqual(cache.has_value(word), True)
            self.assertEqual(cache.get_value(word), special_meanings[word])

        self.assertEqual(cache.get_cached_keys(), ["there", "is", "no", "sense"])

        for word in additions2:
            if cache.has_value(word):
                self.assertEqual(cache.get_value(word), special_meanings[word])
            else:
                cache.set_value(word, special_meanings[word])

        self.assertEqual(cache.get_cached_keys(), ["there", "we", "is", "no"], "we jumps")

        cache.get_value("is")
        self.assertEqual(cache.get_cached_keys(), ["there", "we", "is", "no"], "is is there 2 times")

        cache.get_value("is")
        self.assertEqual(cache.get_cached_keys(), ["there", "is", "we", "no"], "is is there 3 times")

        cache.get_value("is")
        self.assertEqual(cache.get_cached_keys(), ["there", "is", "we", "no"], "is is there 4 times")

        cache.get_value("is")
        self.assertEqual(cache.get_cached_keys(), ["is", "there", "we", "no"], "is is there 5 times")

        cache.get_value("there")
        cache.get_value("we")
        cache.get_value("we")
        cache.get_value("we")
        cache.get_value("we")
        self.assertEqual(cache.get_cached_keys(), ["we", "is", "there", "no"])


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        current_dir = os.path.dirname(__file__)
        file_path = current_dir + "../../../../resources/embeddings/glove.twitter.27B.50d.enriched.txt"
        file_path = os.path.realpath(file_path)
        print("[1 / 2] Load embeddings")
        cls.in_memory = load_embedding(file_path, True)
        print("[2 / 2] Load embeddings")
        cls.file_based = load_embedding(file_path, False)
        print("Finished loading")

    def test_memory_based_embedding(self):
        # Consistent over multiple requests
        self.assertEqual(
            self.in_memory.get_embedding("hello").tolist(),
            self.in_memory.get_embedding("hello").tolist()
        )

        # Different embeddings have different words
        self.assertNotEqual(
            self.in_memory.get_embedding("hello").tolist(),
            self.in_memory.get_embedding("world").tolist()
        )

    def test_file_based_embedding(self):
        # Consistent over multiple requests
        self.assertEqual(
            self.file_based.get_embedding("hello").tolist(),
            self.file_based.get_embedding("hello").tolist()
        )

        # Different embeddings have different words
        self.assertNotEqual(
            self.file_based.get_embedding("hello").tolist(),
            self.file_based.get_embedding("world").tolist()
        )

    def test_in_memory_embedding(self):
        # Consistent over multiple requests
        self.assertEqual(
            self.in_memory.get_embedding("hello").tolist(),
            self.in_memory.get_embedding("hello").tolist()
        )

        # Different embeddings have different words
        self.assertNotEqual(
            self.in_memory.get_embedding("hello").tolist(),
            self.in_memory.get_embedding("world").tolist()
        )

    def test_both_embedding_consistency(self):
        # Yield the same embeddings
        self.assertEqual(
            self.in_memory.get_embedding("hello").tolist(),
            self.file_based.get_embedding("hello").tolist()
        )
        self.assertEqual(
            self.in_memory.get_embedding("world").tolist(),
            self.file_based.get_embedding("world").tolist()
        )

    def test_missing_in_memory(self):
        # Empty embedding
        self.assertEqual(
            self.in_memory.get_embedding("ujzuegc").tolist(),
            [0.0] * 51
        )

    def test_missing_file_based(self):
        # Empty embedding
        self.assertEqual(
            self.file_based.get_embedding("ujzuegc").tolist(),
            [0.0] * 51
        )

    def test_tweet_extraction(self):
        self.assertEqual(
            len(self.in_memory.get_tweet_embeddings("Hello @potato ujzuegc")),
            3
        )


if __name__ == '__main__':
    unittest.main()
