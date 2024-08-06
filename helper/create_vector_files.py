from src.features import FeatureExtractor
from src.twitter import TweetDataset
import torch
from tqdm import tqdm
from src.features.glove import load_embedding
import numpy as np
import os
import math
from typing import Dict, List
import random

INPUT_DATASET = "../datasets/hatespeech"
OUTPUT_PATH = "../build/vectors/hatespeech"

np.random.seed(42)
random.seed(42)


def permutate_ids_wrt_userid(dataset: TweetDataset):
    user_ids: Dict[str, List[int]] = {}
    for i, entry in enumerate(dataset):
        user_id = entry.post.user_id
        if user_id in user_ids:
            user_ids[user_id].append(i)
        else:
            user_ids[user_id] = [i]

    known_user_indices = []
    remaining_indices = []
    for (key, value) in user_ids.items():
        if len(value) > 1:
            known_user_indices.extend(value)
        else:
            remaining_indices.extend(value)

    assert len(known_user_indices) + len(remaining_indices) == len(dataset)

    random.shuffle(known_user_indices)
    random.shuffle(remaining_indices)

    return [*known_user_indices, *remaining_indices]


os.makedirs(OUTPUT_PATH, exist_ok=True)
if __name__ == "__main__":
    dataset = TweetDataset(INPUT_DATASET, load_user_timeline=False)

    print("Generating splitting file...")
    # First define splitting
    ordered_ids = [str(entry.post.id) for entry in tqdm(dataset)]
    permutation = [str(i) for i in permutate_ids_wrt_userid(dataset)]
    train_set_size = math.floor(len(permutation) * 0.8)
    dev_set_size = math.floor(len(permutation) * 0.1)
    test_set_size = len(permutation) - train_set_size - dev_set_size
    with open(os.path.join(INPUT_DATASET, "splitting.txt"), "w") as file:
        file.writelines(x + "\n" for x in [
            "tweet_ids=" + ",".join(ordered_ids),
            "permutation=" + ",".join(permutation),
            "train=" + ",".join(permutation[:train_set_size]),
            "dev=" + ",".join(permutation[train_set_size:train_set_size + dev_set_size]),
            "test=" + ",".join(permutation[train_set_size + dev_set_size:]),
            "splitting_percentages=0.8,0.1,0.1",
            "splitting_bucket_size={},{},{}".format(train_set_size, dev_set_size, test_set_size)
        ])

    tweet_text = []
    for d in dataset:
        tweet_text.append(d.post.text)

    torch.save(tweet_text, OUTPUT_PATH + "/x_texts.pt")

    y_data = torch.FloatTensor([x.label for x in dataset])
    torch.save(y_data, OUTPUT_PATH + "/y_data.pt")

    for i in range(1, 8):
        tweet_features = True if i & 1 << 0 else False
        user_features = True if i & 1 << 1 else False
        text_features = True if i & 1 << 2 else False

        f = FeatureExtractor(
            tweet_features=tweet_features,
            user_features=user_features,
            text_features=text_features,
            timeline_features=False
        )

        x_data = []
        for obj in tqdm(dataset, desc="[{}] Extract features".format(i)):
            features = f.get_feature_vector(obj)
            x_data.append(features)

        x_data = torch.FloatTensor(x_data)

        file_name = ""
        if tweet_features:
            file_name += "tweet."
        if user_features:
            file_name += "user."
        if text_features:
            file_name += "text."

        torch.save(x_data, OUTPUT_PATH + "/x_data." + file_name + "pt")

    embedding_files = [
        "25d",
        "25d.enriched",
        "50d",
        "50d.enriched"
    ]

    for embedding in embedding_files:
        print("Load Embeddings: {}".format(embedding))
        glove = load_embedding("resources/embeddings/glove.twitter.27B.{}.txt".format(embedding), keep_in_memory=False)

        x_data_embeddings = []

        for obj in tqdm(dataset, desc="Extract embedding features"):
            x_data_embeddings.append(glove.get_tweet_embeddings(obj.post.text))

        file_name = "." + embedding + ".pt"

        torch.save(x_data_embeddings, OUTPUT_PATH + "/x_embeddings" + file_name)
