"""
enrich-embedding.py

python enrich-embedding.py ../resources/embeddings/glove.twitter.27B.25d.txt

Script to enrich embeddings using sentiment scores. One Dimension will be added representing the
sentiment of that word.
"""

import sys
from tqdm import tqdm
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

if __name__ == "__main__":
    sentiment = SentimentIntensityAnalyzer()
    stemmer = PorterStemmer()

    def get_sentiment(word: str) -> float:
        def get_internal_sentiment(word: str) -> float:
            return sentiment.polarity_scores(word)["compound"]

        score = get_internal_sentiment(word)
        if score == 0.0:
            stemmer.stem(word)
            return get_internal_sentiment(word)
        else:
            return score

    file_name = sys.argv[1]
    to_file_name = None
    if len(sys.argv) > 2:
        to_file_name = sys.argv[2]
    else:
        to_file_name = file_name.replace(".txt", "") + ".enriched.txt"

    with open(file_name, "r", encoding="utf-8") as file_from:
        with open(to_file_name, "w", encoding="utf-8") as file_to:
            sentiments_added = 0
            loop = tqdm(file_from, desc="Adding sentiment scores")
            for line in loop:
                line: str = line.rstrip()
                end_of_word = line.index(" ")
                word = line[:end_of_word]

                score = get_sentiment(word)
                if score != 0.0:
                    sentiments_added += 1
                    loop.set_postfix(enriched=sentiments_added)

                line += " " + str(score) + "\n"

                file_to.write(line)