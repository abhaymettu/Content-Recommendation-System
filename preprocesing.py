import pandas as pd

def load_and_preprocess_data(filepath="user_ratings.csv"):
    df = pd.read_csv(filepath)

    df['user_id'] = df['user_id'].astype("category")
    df['content_id'] = df['content_id'].astype("category")

    user_ids = df['user_id'].cat.codes.values
    content_ids = df['content_id'].cat.codes.values
    ratings = df['rating'].values

    return user_ids, content_ids, ratings
