# ============================================================
# EMOTION DETECTION - FLASK APPLICATION
# ============================================================

from flask import Flask, render_template, request
import tensorflow as tf
import pickle


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. SETTINGS
# ============================================================

MAX_LENGTH = 50

MODEL_PATH = "emotion_lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAPPING_PATH = "emotion_mapping.pkl"


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

with open(
    TOKENIZER_PATH,
    "rb"
) as file:

    tokenizer = pickle.load(file)

print("Tokenizer loaded successfully.")


# ============================================================
# 5. LOAD EMOTION MAPPING
# ============================================================

print("Loading emotion mapping...")

with open(
    MAPPING_PATH,
    "rb"
) as file:

    emotion_to_number = pickle.load(file)

print("Emotion mapping loaded successfully.")


# ============================================================
# 6. CREATE REVERSE MAPPING
# ============================================================

number_to_emotion = {
    number: emotion
    for emotion, number in emotion_to_number.items()
}


print("\nEmotion Mapping:")
print(number_to_emotion)


# ============================================================
# 7. HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # Default values
    text = ""
    emotion = None
    confidence = None
    probabilities = None


    # ========================================================
    # 8. HANDLE FORM SUBMISSION
    # ========================================================

    if request.method == "POST":

        # Get text from HTML form
        text = request.form.get(
            "text",
            ""
        ).strip()


        # ====================================================
        # CHECK EMPTY INPUT
        # ====================================================

        if text:

            # =================================================
            # TOKENIZE
            # =================================================

            sequence = tokenizer.texts_to_sequences(
                [text]
            )


            # =================================================
            # PADDING
            # =================================================

            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(

                sequence,

                maxlen=MAX_LENGTH,

                padding="post",

                truncating="post"
            )


            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(
                padded_sequence,
                verbose=0
            )[0]


            # =================================================
            # FIND BEST EMOTION
            # =================================================

            predicted_number = int(
                tf.argmax(prediction).numpy()
            )


            # Convert number to emotion
            emotion = number_to_emotion[
                predicted_number
            ]


            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = round(
                float(
                    prediction[predicted_number] * 100
                ),
                2
            )


            # =================================================
            # ALL PROBABILITIES
            # =================================================

            probabilities = []

            for number, probability in enumerate(
                prediction
            ):

                probabilities.append({

                    "emotion":
                        number_to_emotion[number],

                    "probability":
                        round(
                            float(probability) * 100,
                            2
                        )
                })


        else:

            # Keep empty values if no text was entered
            emotion = None
            confidence = None
            probabilities = None


    # ========================================================
    # 9. SEND DATA TO HTML
    # ========================================================

    return render_template(

        "index.html",

        text=text,

        emotion=emotion,

        confidence=confidence,

        probabilities=probabilities
    )


# ============================================================
# 10. START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5002
    )
