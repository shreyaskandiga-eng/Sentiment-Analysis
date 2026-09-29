# ============================================================
# EMOTION DETECTION USING LSTM
# ============================================================

import os
import pandas as pd
import tensorflow as tf
import pickle


# ============================================================
# 1. SETTINGS
# ============================================================

VOCAB_SIZE = 10000
MAX_LENGTH = 50
EMBEDDING_DIM = 128
LSTM_UNITS = 128
NUM_EPOCHS = 15
BATCH_SIZE = 32

MODEL_PATH = "emotion_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAPPING_PATH = "emotion_mapping.pkl"


# ============================================================
# 2. RANDOM SEED
# ============================================================

tf.keras.utils.set_random_seed(42)


# ============================================================
# 3. LOAD DATASET
# ============================================================

train_data = pd.read_csv(
    "dataset/train.txt",
    sep=";",
    header=None
)

train_data.columns = ["text", "emotion"]


val_data = pd.read_csv(
    "dataset/val.txt",
    sep=";",
    header=None
)

val_data.columns = ["text", "emotion"]


test_data = pd.read_csv(
    "dataset/test.txt",
    sep=";",
    header=None
)

test_data.columns = ["text", "emotion"]


# Remove missing values
train_data = train_data.dropna()
val_data = val_data.dropna()
test_data = test_data.dropna()


# Remove accidental spaces
train_data["text"] = train_data["text"].astype(str).str.strip()
val_data["text"] = val_data["text"].astype(str).str.strip()
test_data["text"] = test_data["text"].astype(str).str.strip()

train_data["emotion"] = train_data["emotion"].astype(str).str.strip().str.lower()
val_data["emotion"] = val_data["emotion"].astype(str).str.strip().str.lower()
test_data["emotion"] = test_data["emotion"].astype(str).str.strip().str.lower()


print("==========================================")
print("DATASET INFORMATION")
print("==========================================")

print("Training:", train_data.shape)
print("Validation:", val_data.shape)
print("Testing:", test_data.shape)


# ============================================================
# 4. DISPLAY CLASS DISTRIBUTION
# ============================================================

print("\nTraining emotion distribution:")
print(train_data["emotion"].value_counts())

print("\nValidation emotion distribution:")
print(val_data["emotion"].value_counts())

print("\nTesting emotion distribution:")
print(test_data["emotion"].value_counts())


# ============================================================
# 5. FIXED EMOTION MAPPING
# ============================================================

emotion_classes = [
    "anger",
    "fear",
    "joy",
    "love",
    "sadness",
    "surprise"
]


emotion_to_number = {
    emotion: index
    for index, emotion in enumerate(emotion_classes)
}


number_to_emotion = {
    index: emotion
    for emotion, index in emotion_to_number.items()
}


print("\n==========================================")
print("EMOTION MAPPING")
print("==========================================")

print(emotion_to_number)


# ============================================================
# 6. CHECK DATASET LABELS
# ============================================================

unknown_labels = set(train_data["emotion"]) - set(emotion_classes)

if unknown_labels:
    raise ValueError(
        f"Unknown emotion labels found: {unknown_labels}"
    )


# ============================================================
# 7. CREATE TOKENIZER
# ============================================================

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)


tokenizer.fit_on_texts(
    train_data["text"]
)


# ============================================================
# 8. TOKENIZE
# ============================================================

train_sequences = tokenizer.texts_to_sequences(
    train_data["text"]
)

val_sequences = tokenizer.texts_to_sequences(
    val_data["text"]
)

test_sequences = tokenizer.texts_to_sequences(
    test_data["text"]
)


# ============================================================
# 9. PADDING
# ============================================================

train_padded = tf.keras.preprocessing.sequence.pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

val_padded = tf.keras.preprocessing.sequence.pad_sequences(
    val_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

test_padded = tf.keras.preprocessing.sequence.pad_sequences(
    test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


print("\nInput shape:", train_padded.shape)


# ============================================================
# 10. CONVERT LABELS TO NUMBERS
# ============================================================

train_labels = train_data["emotion"].map(
    emotion_to_number
).values

val_labels = val_data["emotion"].map(
    emotion_to_number
).values

test_labels = test_data["emotion"].map(
    emotion_to_number
).values


# ============================================================
# 11. BUILD MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS)
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(emotion_classes),
        activation="softmax"
    )
])


# ============================================================
# 12. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n==========================================")
print("MODEL")
print("==========================================")

model.summary()


# ============================================================
# 13. TRAIN
# ============================================================

print("\n==========================================")
print("TRAINING STARTED")
print("==========================================")


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    train_padded,
    train_labels,

    validation_data=(
        val_padded,
        val_labels
    ),

    epochs=NUM_EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ]
)


# ============================================================
# 14. TEST MODEL
# ============================================================

print("\n==========================================")
print("TESTING MODEL")
print("==========================================")


test_loss, test_accuracy = model.evaluate(
    test_padded,
    test_labels,
    verbose=1
)


print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 15. CHECK WHETHER MODEL PREDICTS ONLY ONE CLASS
# ============================================================

print("\n==========================================")
print("PREDICTION DISTRIBUTION")
print("==========================================")


test_predictions = model.predict(
    test_padded,
    verbose=0
)


predicted_classes = test_predictions.argmax(
    axis=1
)


for number, emotion in number_to_emotion.items():

    count = (predicted_classes == number).sum()

    print(
        f"{number} -> {emotion}: {count}"
    )


# ============================================================
# 16. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\nSaved:", MODEL_PATH)


# ============================================================
# 17. SAVE TOKENIZER
# ============================================================

with open(
    TOKENIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )

print("Saved:", TOKENIZER_PATH)


# ============================================================
# 18. SAVE MAPPING
# ============================================================

with open(
    MAPPING_PATH,
    "wb"
) as file:

    pickle.dump(
        emotion_to_number,
        file
    )

print("Saved:", MAPPING_PATH)


# ============================================================
# 19. FINAL
# ============================================================

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")

print("\nEmotion mapping:")
print(emotion_to_number)

print("\nFinal test accuracy:")
print(round(test_accuracy * 100, 2), "%")
