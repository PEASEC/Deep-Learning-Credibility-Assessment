from typing import Dict, Any, List, Union, Tuple
from datetime import datetime
from .AbstractPost import AbstractPost


class Activity(AbstractPost):
    def __init__(self, activity: Dict[str, Any]):
        self.activity = activity

    @staticmethod
    def from_file(file_name):
        return Activity(AbstractPost._get_json_object(file_name))

    @property
    def object(self):
        return self.activity["object"]

    @property
    def actor(self):
        return self.activity["actor"]

    @property
    def id(self) -> str:
        return self.object["@id"]

    @property
    def text(self) -> str:
        result = self.object["content"]

        # Fix special characters
        result = result.replace("&amp;", "&")
        result = result.replace("&quot;", "\"")
        result = result.replace("&gt;", ">")
        result = result.replace("&lt;", "<")
        return result

    @property
    def create_date(self) -> datetime:
        return datetime.fromtimestamp(self.object["startTime"] // 1000)

    @property
    def user_status_count(self) -> int:
        return self.actor["numStatuses"]

    @property
    def user_friends_count(self) -> int:
        return self.actor["numFriends"]

    @property
    def user_listed_count(self) -> int:
        return self.actor["numListed"]

    @property
    def user_followers_count(self) -> int:
        return self.actor["numFollowers"]

    @property
    def user_favourites_count(self) -> int:
        return self.actor["numFavorites"]

    @property
    def user_screen_name(self) -> str:
        return self.actor["userName"]

    @property
    def user_id(self) -> str:
        return self.actor["@id"]

    @property
    def user_name(self) -> str:
        return self.actor["displayName"]

    @property
    def user_create_date(self) -> datetime:
        return datetime.fromtimestamp(self.actor["profileCreatedAt"] // 1000)

    @property
    def user_verified(self) -> bool:
        return self.actor["verified"]

    @property
    def user_profile_picture(self) -> Union[str, None]:
        if "profileImage" in self.actor:
            url = self.actor["profileImage"]
            return url if url != "" else None
        return None

    @property
    def user_profile_banner_url(self) -> Union[str, None]:
        if "profileBackground" in self.actor:
            profile_banner = self.actor["profileBackground"]
            return profile_banner if profile_banner is not None and profile_banner != "" else None
        return None

    @property
    def user_description(self) -> Union[str, None]:
        if "content" in self.actor:
            return self.actor["content"]
        else:
            return None

    @property
    def user_location(self) -> Union[str, None]:
        if "location" in self.actor:
            location = self.actor["location"]
            if "displayName" in location:
                display_name = location["displayName"]
                if display_name is not None and display_name != "":
                    return display_name

        return None

    @property
    def user_urls(self) -> List[str]:
        urls = []
        if "homePage" in self.actor:
            urls.append(self.actor["homePage"])

        description = self.user_description
        if description is not None:
            urls.extend(AbstractPost._extract_urls(description))

        return urls

    @property
    def urls(self) -> List[str]:
        enriched_data = self.object["enrichedData"]
        if "embeddedUrls" in enriched_data:
            return enriched_data["embeddedUrls"]
        else:
            return AbstractPost._extract_urls(self.text)

    @property
    def hashtags(self) -> List[str]:
        enriched = self.object["enrichedData"]
        if "tags" in enriched:
            return enriched["tags"]
        else:
            return []

    @property
    def user_mentions(self) -> List[Tuple[str, int]]:
        enriched = self.object["enrichedData"]
        if "mentions" in enriched:
            return enriched["mentions"]
        else:
            return []

    @property
    def media(self) -> List[Tuple[str, str]]:
        enriched = self.object["enrichedData"]
        if "media" in enriched:
            return [m["url"] for m in enriched["media"]]
        else:
            return []

    @property
    def source(self) -> str:
        return self.object["source"]

    @property
    def lang(self) -> str:
        return self.object["enrichedData"]["language"]

    @property
    def is_reply(self) -> bool:
        return self.activity["inReplyTo"] is not None
