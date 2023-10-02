import pandas as pd
import tensorflow as tf

# Load data from CSV
def load_and_preprocess_data(filepath="data/user_ratings.csv"):
    df = pd.read_csv(filepath)

    # Encode user_id and content_id as category types
    df['user_id'] = df['user_id'].astype("category")
    df['content_id'] = df['content_id'].astype("category")

    # Create a matrix of user_id, content_id, and rating
    user_ids = df['user_id'].cat.codes.values
    content_ids = df['content_id'].cat.codes.values
    ratings = df['rating'].values

    return user_ids, content_ids, ratings

# Load data
user_ids, content_ids, ratings = load_and_preprocess_data()

# Model Parameters
EMBEDDING_SIZE = 50
NUM_USERS = len(set(user_ids))
NUM_CONTENTS = len(set(content_ids))

# Collaborative filtering model using embeddings
user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
content_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='content_input')

user_embedding = tf.keras.layers.Embedding(NUM_USERS, EMBEDDING_SIZE)(user_input)
content_embedding = tf.keras.layers.Embedding(NUM_CONTENTS, EMBEDDING_SIZE)(content_input)

dot_product = tf.keras.layers.Dot(axes=2)([user_embedding, content_embedding])
dot_product = tf.keras.layers.Flatten()(dot_product)

model = tf.keras.Model(inputs=[user_input, content_input], outputs=dot_product)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([user_ids, content_ids], ratings, epochs=10, batch_size=64, validation_split=0.2)

# Save the model
model.save('models/recommendation_model.h5')

print("Model trained and saved successfully!")
