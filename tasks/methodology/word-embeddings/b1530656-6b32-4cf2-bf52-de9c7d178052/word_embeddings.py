import warnings
import sys

import pandas as pd
from gensim.models import Word2Vec
import string

import mlflow
import mlflow.sklearn


def clean_doc(doc):
    """
    Cleaning the document before vectorization.
    """
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read command line inputs
    word1 = str(sys.argv[1]) if len(sys.argv) > 1 else "fbi"
    word2 = str(sys.argv[2]) if len(sys.argv) > 2 else "nypd"
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    with mlflow.start_run():
        # Read the fake data csv file using public s3 url
        url = "https://shreyasjothish.s3.us-east-2.amazonaws.com/fake.csv"
        df = pd.read_csv(url)

        # The information related to document is contained in title and text
        # columns. So I am using only these two columns.
        df['title_text'] = df['title'] + df['text']
        df.drop(columns=['uuid', 'ord_in_thread', 'author', 'published',
                         'title', 'text', 'language', 'crawled', 'site_url',
                         'country', 'domain_rank', 'thread_title',
                         'spam_score', 'main_img_url', 'replies_count',
                         'participants_count', 'likes', 'comments', 'shares',
                         'type'], inplace=True)
        df.dropna(inplace=True)
        df.title_text = df.title_text.str.lower()

        # Turn a document into clean tokens.
        df['cleaned'] = df.title_text.apply(clean_doc)
        print(df.shape)

        mlflow.log_param("word1", word1)
        mlflow.log_param("word2", word2)
        mlflow.log_param("topn", topn)

        # Build the model using gensim.
        w2v = Word2Vec(df.cleaned, min_count=20, window=3,
                       size=300, negative=20)
        words = list(w2v.wv.vocab)
        mlflow.log_metric("vocabulary_size", len(words))

        # Explore the results like finding most similar words and similarity
        word1_most_similar = w2v.wv.most_similar(word1, topn=topn)
        print(word1_most_similar)

        word2_most_similar = w2v.wv.most_similar(word2, topn=topn)
        print(word2_most_similar)

        similarity_score = w2v.wv.similarity(word1, word2)
        mlflow.log_metric("similarity_score", similarity_score)
