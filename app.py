import gradio as gr
import tensorflow as tf
import pickle
from keras.layers import LSTM, Bidirectional

# ==== CONFIG ====
MODEL_PATH = "Bidirectional_LSTM.h5"
TOKENIZER_PATH = "tokenizer.pkl"
SPACE_URL = "https://huggingface.co/spaces/Saichandu47/movie_review_LSTM"
MAX_LEN = 750  # Must match training

# ==== LOAD MODEL ====
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"LSTM": LSTM, "Bidirectional": Bidirectional}
)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ==== PREDICTION FUNCTION ====
def predict_sentiment(review_text):
    review_text = review_text.strip()
    if not review_text:
        return "Please enter a review."

    sequence = tokenizer.texts_to_sequences([review_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_LEN)

    proba = model.predict(padded, verbose=0)[0][0]
    label = "Positive" if proba >= 0.5 else "Negative"
    confidence = proba if proba >= 0.5 else (1 - proba)

    return f"{label} ({confidence * 100:.2f}% confidence)"

# ==== UI ====
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="gray")) as demo:
    gr.Markdown(
        """
        # 🎬 Movie Review Sentiment (BiLSTM)
        Fine-tuned BiLSTM model. Predicts Positive/Negative sentiment with confidence.
        """
    )

    with gr.Row():
        with gr.Column():
            review = gr.Textbox(
                label="Enter a movie review",
                placeholder="Type your review here...",
                lines=6
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Prediction", interactive=False)
            gr.HTML(f"""
                <a href="{SPACE_URL}" target="_blank" style="
                    display: inline-flex; align-items: center; gap: 6px;
                    text-decoration: none; padding: 10px 15px;
                    background-color: #444; color: white; border-radius: 6px;">
                    🔗 Share via Link
                </a>
            """)

    examples = [
        "I absolutely loved this movie. The acting and the plot were fantastic!",
        "The film was boring and way too long. I wouldn't recommend it.",
        "It was okay, not the best, not the worst."
    ]
    gr.Examples(examples=examples, inputs=review)

    # Button events
    submit_btn.click(predict_sentiment, inputs=review, outputs=output)
    clear_btn.click(lambda: "", outputs=review)

demo.launch()
