from fastapi import FastAPI, HTTPException
from typing import List, Dict, Union
import uvicorn
from threading import Lock
from src.ml.execute import Executable, FeatureTypes, BertExecutable, RnnExecutable
from src.twitter import Activity, Tweet, AbstractPost, PostObject
import torch
import os

TRUE_WORDS = ["true", "on", "yes", "1"]

class DescriptionObject:
    def __init__(self, description: str):
        self.description = description


class ModelInfo(DescriptionObject):
    def __init__(self, description: str, executable: Executable):
        super().__init__(description)
        self.executable = executable

    def to_base(self) -> DescriptionObject:
        return DescriptionObject(self.description)

print("Loading models and resources...")

available_models: Dict[str, ModelInfo] = {}

available_models["hatespeech"] = ModelInfo(
    description="A classifier to detect hatespeech based on recurrent neural networks.",
    executable=RnnExecutable(
        "build/hatespeech", FeatureTypes.TEXT_TWEET_USER
    )
)

if torch.cuda.is_available() and os.getenv("IGNORE_GPU", "False").lower() not in TRUE_WORDS:
    available_models["credibility"] = ModelInfo(
        description="A classifier to detect credibility based on the BERT language model.",
        executable=BertExecutable(
            "build/credibility",
            "./resources/bert/credibility",
            FeatureTypes.TEXT_TWEET_USER
        )
    )
else:
    available_models["credibility"] = ModelInfo(
        description="A classifier to detect credibility based on recurrent neural networks. (No GPU available)",
        executable=RnnExecutable(
            "build/credibility",
            FeatureTypes.TEXT_TWEET_USER
        )
    )

shall_initialize_all = os.getenv("INITIALIZE_ALL_ON_STARTUP", "")
if shall_initialize_all.lower() in TRUE_WORDS:
    # Load dummy tweet
    from src.twitter import Tweet, PostObject
    tweet = Tweet.from_file("resources/dummy_tweet.json")
    post = PostObject(tweet, {}, [])

    for key in available_models:
        available_models[key].executable.predict([post])

print("Done loading")

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/classifiers")
async def available_classifiers():
    result: Dict[str, DescriptionObject] = {}
    for (key, value) in available_models.items():
        result[key] = value.to_base()

    return result


def classify_posts(classifier: str, posts: List[PostObject]):
    if classifier not in available_models:
        raise HTTPException(status_code=404, detail="Unknown classifier name '{}'.".format(classifier))
    else:
        return available_models[classifier].executable.predict(posts).tolist()


# Force parallel execution
mylock = Lock()


@app.post("/classify/{classifier}")
def classify(classifier: str, items: List[dict], data_type: str = "activity") -> Union[List[str], List[List[str]]]:
    posts: List[AbstractPost]
    if data_type == "activity":
        posts = [Activity(p) for p in items]
    elif data_type == "tweet":
        posts = [Tweet(p) for p in items]
    else:
        raise HTTPException(status_code=400, detail="Unknown input type information '{}'.".format(data_type))

    post_objects = [PostObject(p) for p in posts]

    with mylock:
        print("Executing in parallel")

        if "," in classifier:
            classifier = [c.strip() for c in classifier.split(",")]
            return [classify_posts(c, post_objects) for c in classifier]
        else:
            return classify_posts(classifier, post_objects)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("CLASSIFIER_HOST", "0.0.0.0"),
        port=int(os.getenv("CLASSIFIER_PORT", "8000"))
    )
