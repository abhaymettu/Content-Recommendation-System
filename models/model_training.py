import pandas as pd
import tensorflow as tf

def load_and_preprocess_data(filepath="data/user_ratings.csv"):
    df = pd.read_csv(filepath)

    df['user_id'] = df['user_id'].astype("category")
    df['content_id'] = df['content_id'].astype("category")

    user_ids = df['user_id'].cat.codes.values
    content_ids = df['content_id'].cat.codes.values
    ratings = df['rating'].values

    return user_ids, content_ids, ratings

user_ids, content_ids, ratings = load_and_preprocess_data()

EMBEDDING_SIZE = 50
NUM_USERS = len(set(user_ids))
NUM_CONTENTS = len(set(content_ids))

user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
content_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='content_input')

user_embedding = tf.keras.layers.Embedding(NUM_USERS, EMBEDDING_SIZE)(user_input)
content_embedding = tf.keras.layers.Embedding(NUM_CONTENTS, EMBEDDING_SIZE)(content_input)

dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, content_embedding])
dot_product = tf.keras.layers.Flatten()(dot_product)

model = tf.keras.Model(inputs=[user_input, content_input], outputs=dot_product)

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit([user_ids, content_ids], ratings, epochs=10, batch_size=64, validation_split=0.2)

# Save the model
model.save('models/recommendation_model.h5')

print("Model trained and saved successfully!")
