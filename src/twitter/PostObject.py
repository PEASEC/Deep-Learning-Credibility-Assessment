from typing import Dict, List
from .AbstractPost import AbstractPost


class PostObject:
    def __init__(
            self,
            post: AbstractPost,
            user_profile: Dict[str, str] = None,
            user_timeline_posts: List[AbstractPost] = None,
    ):
        self.post = post
        self.user_profile = user_profile
        self.user_timeline_posts = user_timeline_posts if user_timeline_posts is not None else []


class LabeledPostObject(PostObject):
    def __init__(
            self,
            post: AbstractPost,
            label: float,
            source: str,
            user_profile: Dict[str, str] = None,
            user_timeline_posts: List[AbstractPost] = None,
    ):
        super(LabeledPostObject, self).__init__(post, user_profile, user_timeline_posts)
        self.label = label
        self.source = source

    def remove_label(self) -> PostObject:
        return PostObject(self.post, self.user_profile, self.user_timeline_posts)
