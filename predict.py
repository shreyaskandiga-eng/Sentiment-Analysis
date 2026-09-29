# ============================================================
# EMOTION DETECTION - PREDICTION
# ============================================================

import tensorflow as tf
import pickle


# ============================================================
# SETTINGS
# ============================================================

MAX_LENGTH = 50


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    "emotion_lstm_model.keras"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

with open(
    "tokenizer.pkl",
    "rb"
) as file:

    tokenizer = pickle.load(file)


# ============================================================
# LOAD MAPPING
# ============================================================

with open(
    "emotion_mapping.pkl",
    "rb"
) as file:

    emotion_to_number = pickle.load(file)


number_to_emotion = {
    number: emotion
    for emotion, number in emotion_to_number.items()
}


# ============================================================
# DISPLAY MAPPING
# ============================================================

print("\nEmotion Mapping:")
print(number_to_emotion)


# ============================================================
# LOOP
# ============================================================

while True:

    text = input(
        "\nEnter a sentence (or type quit): "
    )


    if text.lower() == "quit":
        break


    # ========================================================
    # TOKENIZE
    # ========================================================

    sequence = tokenizer.texts_to_sequences(
        [text]
    )


    # ========================================================
    # PADDING
    # ========================================================

    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )


    # ========================================================
    # PREDICT
    # ========================================================

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0]


    # ========================================================
    # DISPLAY ALL PROBABILITIES
    # ========================================================

    print("\n-----------------------------")

    for number, probability in enumerate(prediction):

        print(
            f"{number_to_emotion[number]:10s} : "
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # BEST CLASS
    # ========================================================

    predicted_number = int(
        tf.argmax(prediction).numpy()
    )


    predicted_emotion = number_to_emotion[
        predicted_number
    ]


    confidence = (
        prediction[predicted_number] * 100
    )


    print("-----------------------------")

    print(
        "Predicted Emotion:",
        predicted_emotion
    )

    print(
        "Confidence:",
        round(float(confidence), 2),
        "%"
    )

    print("-----------------------------")