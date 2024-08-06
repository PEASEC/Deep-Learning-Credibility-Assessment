import numpy as np
from src.twitter import AbstractPost
from src.features.WordList import FamousNameIndex, SurnameIndex, Pronouns, TopWordIndex, ProfanityWordIndex
from src.features.DateList import HolidayIndex
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import string
from typing import Tuple, List, cast, Any, Sized
import emoji


def location_is_meaningful(location: str) -> bool:
    """
    Determines if a location is most likely to be meaningful:
    'New York, NY' is meaningful
    'All around the world' is not meaningful
    """
    compass = ["south", "north", "east", "west", "lower", "upper"]
    jokes = [
        "hell", "internet", "world", "everywhere", "global",
        "twitter", "reddit", "4chan", "facebook", "instagram",
        "earth"
    ]

    if max([location.lower().find(w) for w in jokes]) >= 0:
        return False

    score = 0
    if max([max(location.lower().find(w), location.lower().find(w + "ern")) for w in compass]) >= 0:
        score += 1

    parts = location.split(",")
    if len(parts) == 2:
        # String is constructed as "COUNTRY, STATE"
        score += 1
        if len(parts[1].strip()) <= 3:
            # State might be a short form, e.g GER, DE, USA, TX (Texas)
            score += 1

    parts = location.split(" ")
    if len(parts) <= 3:
        # Shorter forms are more likely true
        score += 1

    for p in parts:
        p = p.strip()
        if p in ["and", "/"] or len(p) > 1 and p[0].isupper() and p[1].islower():
            score += 2.0 / len(parts)

    parts = location.split(".")
    if max([len(p) for p in parts]) <= 3:
        # Location consists of abbreviation
        score += 1

    return score >= 1.99


def remove_entities_from_text(text: str):
    http_chars = "/._-@:%&?#"
    result = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char == '@' or char == "#":
            while (i + 1) < len(text) and text[i + 1].isalnum():
                i += 1
        elif char == "h" and (text[i:i + 7] == "http://" or text[i:i + 8] == "https://"):
            i += 7
            while (i + 1) < len(text) and (text[i + 1].isalnum() or text[i + 1] in http_chars):
                i += 1
        else:
            result += char

        i += 1
    return result


def normalize(func):
    def wrapper(*args, **kwargs):
        return np.log(func(*args, **kwargs) + 1)

    return wrapper


def len_or_one(value: Sized):
    result = len(value)
    if result <= 0:
        return 1
    else:
        return float(result)


class TweetFeatures:
    def __init__(self, tweet: AbstractPost):
        self.tweet = tweet

    @staticmethod
    def initialize():
        HolidayIndex.initialize()

    @property
    def day_of_week(self):
        return self.tweet.create_date.weekday()

    @property
    def quarter_of_day(self):
        return int(self.tweet.create_date.hour / 4)

    @property
    def published_on_holiday(self):
        return 1 if HolidayIndex.is_holiday(self.tweet.create_date.date()) else 0

    @property
    def urls_count(self):
        return len(self.tweet.urls)

    @property
    def avg_len_urls(self):
        """
        Intuition: Short urls only point on domains (e.g. https://www.nytimes.com/)
        while longer urls point to a specific article (e.g.
        https://www.nytimes.com/2020/03/31/movies/coronavirus-movie.html)
        """
        urls = self.tweet.urls
        if len(urls) <= 0:
            return 0
        else:
            return np.average([len(u) for u in self.tweet.urls])

    @property
    def mentions_count(self):
        return len(self.tweet.user_mentions)

    @property
    def hashtag_count(self):
        return len(self.tweet.hashtags)

    @property
    def avg_hashtag_length(self):
        lengths = [len(h) for h in self.tweet.hashtags]
        if len(lengths) == 0:
            return 0
        else:
            return np.average(lengths)

    @property
    def media_count(self):
        return len(self.tweet.media)

    @property
    def source(self):
        # Ordered from Mobile to Desktop and Free/Default to Professional/Enterprise solution
        clients = ["Twitter for Android", "Twitter for iPhone", "Twitter for iPad", "Twitter Web Client", "TweetDeck",
                   "IFTTT", "twitterfeed", "dlvr.it", "HootSuite", "Spinklr", "Buffer", "SocialFlow"]
        source = self.tweet.source
        return clients.index(source) if source in clients else len(clients)


class UserFeatures:
    def __init__(self, tweet: AbstractPost):
        self.tweet = tweet

    @staticmethod
    def initialize():
        FamousNameIndex.initialize()
        SurnameIndex.initialize()

    @property
    def user_is_verified(self):
        return 1 if self.tweet.user_verified else 0

    @property
    @normalize
    def user_account_age(self):
        """
        Account age relative to the age of the post
        """
        delta = self.tweet.create_date - self.tweet.user_create_date
        return delta.days

    @property
    def user_location_is_set(self):
        return 1 if self.tweet.user_location is not None else 0

    @property
    def user_location_score(self):
        """
        Returns:
            0 if no location is set
            1 if a location is set but not meaningful
            2 if a location is set and meaningful
        """
        if self.user_location_is_set == 1:
            return 2 if location_is_meaningful(self.tweet.user_location) else 1
        else:
            return 0

    @property
    @normalize
    def user_status_count(self):
        return self.tweet.user_status_count

    @property
    @normalize
    def user_friends_count(self):
        return self.tweet.user_friends_count

    @property
    @normalize
    def user_followers_count(self):
        return self.tweet.user_followers_count

    @property
    @normalize
    def user_listed_count(self):
        return self.tweet.user_listed_count

    @property
    @normalize
    def user_favorite_count(self):
        return self.tweet.user_favourites_count

    @property
    def user_profile_picture_is_set(self):
        if self.tweet.user_profile_picture is None:
            return 1

        return not self.tweet.user_profile_picture.endswith("default_profile_normal.png")

    @property
    def user_profile_banner_is_set(self):
        return 0 if self.tweet.user_profile_banner_url is None else 1

    @property
    def user_url_count(self):
        return len(self.tweet.user_urls)

    @property
    def user_name_length(self):
        return len(self.tweet.user_screen_name)

    @property
    def user_name_contains_only_alpha(self):
        return 1 if self.tweet.user_screen_name.replace(" ", "").isalpha() else 0

    @property
    def user_full_name_is_real_name(self):
        name = self.tweet.user_name
        if name.find(",") >= 0:
            name = name.split(",")[1].strip()

        surname = name.split(" ")[0]
        return 1 if SurnameIndex.is_name(surname) else 0

    @property
    def user_full_name_is_famous(self):
        return 1 if FamousNameIndex.is_famous_name(self.tweet.user_name) else 0

    @property
    def user_description_length(self):
        if self.tweet.user_description is None:
            return 0
        else:
            return len(self.tweet.user_description)


class TextFeatures:
    def __init__(self, tweet: AbstractPost):
        self.tweet = tweet
        self.tokens = word_tokenize(self.tweet.text)

    STOP_WORDS: set = None
    PUNCTUATION: set = set(string.punctuation)

    sentiment_intensity_analyzer: SentimentIntensityAnalyzer = None

    @staticmethod
    def initialize():
        TopWordIndex.initialize()
        Pronouns.initialize()
        ProfanityWordIndex.initialize()

        nltk.download('vader_lexicon')
        nltk.download('punkt')
        nltk.download('stopwords')

        TextFeatures.STOP_WORDS = set(stopwords.words("english"))
        TextFeatures.sentiment_intensity_analyzer = SentimentIntensityAnalyzer()

    @property
    def text_length(self):
        return len(self.tweet.text)

    @property
    def text_unique_characters_count(self):
        return len(set(self.tweet.text)) / len_or_one(self.tweet.text)

    @property
    def text_alpha_character_count(self):
        return len([x for x in self.tweet.text if x.isalpha()]) / len_or_one(self.tweet.text)

    @property
    def text_num_character_count(self):
        return len([x for x in self.tweet.text if x.isnumeric()]) / len_or_one(self.tweet.text)

    @property
    def text_uppercase_character_count(self):
        return len([x for x in self.tweet.text if x.isupper()]) / len_or_one(self.tweet.text)

    @property
    def text_words_count(self):
        words = self.tokens
        return len([word for word in words if word.isalpha()]) / len_or_one(words)

    @property
    def text_uppercase_word_count(self):
        words = self.tokens
        return len([word for word in words if word.isalpha() and word.isupper()]) / len_or_one(words)

    @property
    def text_stop_words_count(self):
        words = self.tokens
        return len([word for word in words if word in TextFeatures.STOP_WORDS]) / len_or_one(words)

    @property
    def text_avg_word_len(self):
        word_lengths = [len(word) for word in self.tokens if word.isalpha()]
        if len(word_lengths) <= 0:
            return 0
        else:
            return np.average(word_lengths)

    @property
    def text_exclamation_mark_count(self):
        return self.tweet.text.count("!") / len_or_one(self.tweet.text)

    @property
    def text_question_mark_count(self):
        return self.tweet.text.count("?") / len_or_one(self.tweet.text)

    @property
    def text_dot_count(self):
        return self.tweet.text.count(".") / len_or_one(self.tweet.text)

    @property
    def text_sentiment(self) -> Tuple[float, float, float]:
        result = self.sentiment_intensity_analyzer.polarity_scores(self.tweet.text)
        return result["neg"], result["neu"], result["pos"]

    @property
    def text_pronouns(self) -> Tuple[int, int, int, int, int, int]:
        result: List[int] = [0] * 6
        words = self.tokens
        for word in words:
            pronoun_type = Pronouns.pronoun_type(word)
            if pronoun_type is not None:
                person, plural = pronoun_type
                index = person - 1
                if plural:
                    index += 3

                result[index] += 1

        result = np.array(result) / len_or_one(words)
        return cast(Any, tuple(result))

    @property
    def text_contains_quote(self) -> bool:
        text = self.tweet.text
        complex_double_quotation_marks = text.count(chr(131)) + text.count(chr(120))
        if complex_double_quotation_marks >= 2:
            return True

        double_quotation = text.count('"')
        if double_quotation % 2 == 0 and double_quotation > 0:
            return True

        # Replace complex single quotation marks (may be used as apostrophe)
        text = text.replace(chr(8217), "'").replace(chr(8216), "'")

        single_quotation_left = text.count(" '")
        single_quotation_right = text.count("' ")
        if single_quotation_left == single_quotation_right and single_quotation_left > 0:
            return True

        return False

    @property
    def text_emoticon_count(self):
        return len(emoji.emoji_lis(self.tweet.text)) / len_or_one(self.tweet.text)

    @property
    def text_words_not_in_top_10k_count(self):
        words = word_tokenize(remove_entities_from_text(self.tweet.text))
        words = [w for w in words if w not in string.punctuation]

        return len([w for w in words if TopWordIndex.is_top_word(w)]) / len_or_one(words)

    @property
    def text_profanity_word_count(self):
        words = self.tokens

        return len([w for w in words if ProfanityWordIndex.is_profanity_word(w)]) / len_or_one(words)
