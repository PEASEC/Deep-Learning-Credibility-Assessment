from typing import List, Union, Tuple
import json
from datetime import datetime
from abc import abstractmethod
import re


class AbstractPost:
    @staticmethod
    def _get_json_object(file):
        with open(file) as file_ptr:
            return json.load(file_ptr)

    @staticmethod
    def _extract_urls(input_url: str) -> List[str]:
        groups = re.compile(r"(https?|ftp)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?") \
            .findall(input_url)
        return [
            "{}/{}{}".format(protocol, domain, path)
            for (protocol, domain, path) in groups
            if domain != "t.co"
        ]

    @property
    @abstractmethod
    def id(self) -> str:
        ...

    @property
    @abstractmethod
    def text(self) -> str:
        ...

    @property
    @abstractmethod
    def create_date(self) -> datetime:
        ...

    @property
    @abstractmethod
    def user_status_count(self) -> int:
        ...

    @property
    @abstractmethod
    def user_friends_count(self) -> int:
        ...

    @property
    @abstractmethod
    def user_listed_count(self) -> int:
        ...

    @property
    @abstractmethod
    def user_followers_count(self) -> int:
        ...

    @property
    @abstractmethod
    def user_favourites_count(self) -> int:
        ...

    @property
    @abstractmethod
    def user_screen_name(self) -> str:
        """
        The internal name of the user. For example "@realdonaldtrump"
        """
        ...

    @property
    @abstractmethod
    def user_id(self) -> str:
        ...

    @property
    @abstractmethod
    def user_name(self) -> str:
        """
        The displayed name of the user. For example "Donald J. Trump"
        """
        ...

    @property
    @abstractmethod
    def user_create_date(self) -> datetime:
        ...

    @property
    @abstractmethod
    def user_verified(self) -> bool:
        ...

    @property
    @abstractmethod
    def user_profile_picture(self) -> Union[str, None]:
        ...

    @abstractmethod
    def user_profile_banner_url(self) -> Union[str, None]:
        ...

    @property
    @abstractmethod
    def user_description(self) -> Union[str, None]:
        ...

    @property
    @abstractmethod
    def user_location(self) -> Union[str, None]:
        ...

    @property
    @abstractmethod
    def user_urls(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def urls(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def hashtags(self) -> List[str]:
        ...

    @property
    @abstractmethod
    def user_mentions(self) -> List[Tuple[str, int]]:
        ...

    @property
    @abstractmethod
    def media(self) -> List[Tuple[str, str]]:
        ...

    @property
    @abstractmethod
    def source(self) -> str:
        ...

    @property
    @abstractmethod
    def lang(self) -> str:
        ...

    @property
    @abstractmethod
    def is_reply(self) -> bool:
        ...
