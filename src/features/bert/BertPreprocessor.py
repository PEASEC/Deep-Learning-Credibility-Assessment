from typing import Union
import re
import emoji


def preprocess_bert(
        tweet: str,
        url_token: Union[str, None] = "[URL]",
        mention_token: Union[str, None] = "[MENTION]",
        hashtag_token: Union[str, None] = None,
        allcaps_token: Union[str, None] = None,
        replace_emojis: bool = True,
        replace_smileys: bool = False
):
    if url_token is not None:
        tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", url_token, tweet)

    if mention_token is not None:
        # https://help.twitter.com/en/managing-your-account/twitter-username-rules
        tweet = re.sub(r"@[A-Za-z0-9_]+", mention_token, tweet)

    if hashtag_token is not None:
        tweet = re.sub(r"\#[A-Za-z0-9_]+", hashtag_token, tweet)

    if allcaps_token is not None:
        words = re.findall(r"([A-Z]{2,})", tweet)
        for word in set(words):
            tweet = tweet.replace(word, allcaps_token + " " + word.lower())

    if replace_emojis:
        tweet = emoji.demojize(tweet, True)

    if replace_smileys:
        tweet = tweet.replace("<3", ":red_heart:")
        tweet = tweet.replace(":)", ":smile:")
        tweet = tweet.replace(":D", ":grin:")
        tweet = tweet.replace(":/", ":confused_face:")
        tweet = tweet.replace(":(", ":frowning_face:")
        tweet = tweet.replace(":'(", ":cry:")

    return tweet

