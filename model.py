import tensorflow as tf
from preprocessing import load_and_preprocess_data

# Load data
user_ids, content_ids, ratings = load_and_preprocess_data()

# Model Parameters
EMBEDDING_SIZE = 50
NUM_USERS = len(set(user_ids))
NUM_CONTENTS = len(set(content_ids))
DROPOUT_RATE = 0.2
L2_REG = 0.001

# Model
user_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='user_input')
content_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name='content_input')

user_embedding = tf.keras.layers.Embedding(NUM_USERS, EMBEDDING_SIZE, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG))(user_input)
content_embedding = tf.keras.layers.Embedding(NUM_CONTENTS, EMBEDDING_SIZE, embeddings_regularizer=tf.keras.regularizers.l2(L2_REG))(content_input)

concat = tf.keras.layers.Concatenate()([user_embedding, content_embedding])
concat = tf.keras.layers.Flatten()(concat)

dense = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(concat)
dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(dense)
dense = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(dropout)
dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(dense)
output = tf.keras.layers.Dense(1)(dropout)

model = tf.keras.Model(inputs=[user_input, content_input], outputs=output)

# Learning Rate Scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='mean_squared_error')

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Train the model
model.fit([user_ids, content_ids], ratings, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping, checkpoint, tensorboard])

model.save('recommendation_model.h5')
