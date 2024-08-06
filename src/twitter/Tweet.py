from .AbstractPost import AbstractPost
from typing import List, Tuple, Dict, Any, Union
from datetime import datetime
import re

SOURCE_REGEX = re.compile(r"<a .+>(.+?)</a>")

class Tweet(AbstractPost):
    def __init__(self, tweet: Dict[str, Any]):
        self.tweet = tweet

    @staticmethod
    def from_file(file_name):
        return Tweet(AbstractPost._get_json_object(file_name))

    @property
    def id(self) -> str:
        return self.tweet["id"]

    @property
    def text(self) -> str:
        result: str
        if "full_text" in self.tweet:
            result = self.tweet["full_text"]
        else:
            result = self.tweet["text"]

        # Fix special characters
        result = result.replace("&amp;", "&")
        result = result.replace("&quot;", "\"")
        result = result.replace("&gt;", ">")
        result = result.replace("&lt;", "<")
        return result

    @property
    def create_date(self) -> datetime:
        return datetime.strptime(self.tweet["created_at"], "%a %b %d %H:%M:%S %z %Y")

    @property
    def user_status_count(self) -> int:
        return self.tweet["user"]["statuses_count"]

    @property
    def user_friends_count(self) -> int:
        return self.tweet["user"]["friends_count"]

    @property
    def user_listed_count(self) -> int:
        return self.tweet["user"]["listed_count"]

    @property
    def user_followers_count(self) -> int:
        return self.tweet["user"]["followers_count"]

    @property
    def user_favourites_count(self) -> int:
        return self.tweet["user"]["favourites_count"]

    @property
    def user_screen_name(self) -> str:
        return self.tweet["user"]["screen_name"]

    @property
    def user_id(self) -> str:
        return self.tweet["user"]["id_str"]

    @property
    def user_name(self) -> str:
        return self.tweet["user"]["name"]

    @property
    def user_create_date(self) -> datetime:
        return datetime.strptime(self.tweet["user"]["created_at"], "%a %b %d %H:%M:%S %z %Y")

    @property
    def user_verified(self) -> bool:
        verified = self.tweet["user"]["verified"]
        if verified is None:
            return False

        return verified

    @property
    def user_profile_picture(self) -> Union[str, None]:
        url = self.tweet["user"]["profile_image_url_https"]
        if url is None:
            url = self.tweet["user"]["profile_image_url"]
        return url if url != "" else None

    @property
    def user_profile_banner_url(self) -> Union[str, None]:
        user = self.tweet["user"]
        if "profile_banner_url" in user:
            profile_banner = user["profile_banner_url"]
            return profile_banner if profile_banner != "" else None
        return None

    @property
    def user_description(self) -> Union[str, None]:
        return self.tweet["user"]["description"]

    @property
    def user_location(self) -> Union[str, None]:
        location = self.tweet["user"]["location"]
        return location if len(location) > 0 else None

    @property
    def user_urls(self) -> List[str]:
        urls = []
        entities = self.tweet["user"]["entities"]
        if "url" in entities:
            profile_urls = entities["url"]["urls"]
            urls.extend(profile_urls)

        if "description" in entities:
            description_urls = entities["description"]["urls"]
            urls.extend(description_urls)

        return [u["expanded_url"] for u in urls]

    @property
    def urls(self) -> List[str]:
        entity_urls: List[Dict[str, str]] = self.tweet["entities"]["urls"]
        if len(entity_urls) <= 0:
            return AbstractPost._extract_urls(self.text)
        else:
            return [url["expanded_url"] for url in entity_urls]

    @property
    def hashtags(self) -> List[str]:
        hashtags: List[Dict[str, str]] = self.tweet["entities"]["hashtags"]
        return [h["text"] for h in hashtags]

    @property
    def user_mentions(self) -> List[Tuple[str, int]]:
        mentions = self.tweet["entities"]["user_mentions"]
        return [(m["screen_name"], m["id"]) for m in mentions]

    @property
    def media(self) -> List[Tuple[str, str]]:
        entities = self.tweet["entities"]
        if "media" in entities:
            media = entities["media"]
            return [(m["media_url_https"], m["type"]) for m in media]
        else:
            return []

    @property
    def source(self) -> str:
        source_str = self.tweet["source"]
        match = SOURCE_REGEX.match(source_str)
        if match is not None:
            return match.group(1)
        else:
            return source_str

    @property
    def lang(self) -> str:
        return self.tweet["lang"]

    @property
    def is_reply(self) -> bool:
        return self.tweet["in_reply_to_status_id"] is not None

