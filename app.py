from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertForSequenceClassification
import tensorflow as tf
import streamlit as st
from google.cloud import storage
import os

# Load model and tokenizer
model_path = "distilbert_model_best.h5/tf_model.h5"
config_path_model = "distilbert_model_best.h5/config.json"
# tokenizer_path = "distilbert_tokenizer_best/special_tokens_map.json"
# config_path_tokenizer = "distilbert_tokenizer_best/tokenizer_config.json"

config_model = DistilBertConfig.from_json_file(config_path_model)

# model = TFDistilBertForSequenceClassification.from_pretrained(model_path, config=config_model)

# tokenizer_path = 'distilbert_tokenizer_best'
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# client = storage.Client()
# bucket = client.get_bucket('fake-news-detection-wagon')
# blob = bucket.list_blobs()

# for i in blob:
#     st.write(i.name)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'forward-entity-406417-ec2f9997a2ed.json'

# def load_model_from_bucket(bucket_name, model_path):
# Download the model from Google Cloud Storage
client = storage.Client()
bucket = client.get_bucket('fake-news-detection-wagon')
blob_model = bucket.blob(model_path)
blob_model.download_to_filename('tf_model.h5')
blob_config = bucket.blob(config_path_model)
blob_config.download_to_filename('config.json')
# blob_tokenizer = bucket.blob(tokenizer_path)
# blob_config.download_to_filename(tokenizer_path)
    # Load the model from the downloaded file
model = TFDistilBertForSequenceClassification.from_pretrained('tf_model.h5', config='config.json')
    # return model

# pred_model = load_model_from_bucket('fake-news-detection-wagon', model_path)


# Function to predict using the DistilBERT model
def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="tf")

    # Convert BatchEncoding to Numpy arrays
    input_ids_np = inputs["input_ids"].numpy()
    attention_mask_np = inputs["attention_mask"].numpy()

    # Make prediction
    logits = model.predict({"input_ids": input_ids_np, "attention_mask": attention_mask_np})["logits"]

    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    predicted_class = tf.argmax(logits, axis=1).numpy().item()

    return predicted_class

# Streamlit app
st.title("Fake News Detector")
st.markdown('''
            The technological surge in the past few years has led to a plethora of
            misinformation being spread among the vast corners of the Internet.
            This detector aims to predict whether a given text, namely a news article,
            conveys real information, fake information, or is rather suspicious on the whole.
            ''')
st.markdown('''
            All you need to do is input a text below and we will return an answer,
            as well as the probability.
            ''')

# User input for prediction
user_input = st.text_area("Text input", "Type here...")

if st.button("Predict"):
    # Make prediction when the button is clicked
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        result = "Positive" if sentiment == 1 else "Negative"
        st.success(f"The predicted sentiment is: {result}")
    else:
        st.warning("Please enter text for prediction.")


# txt = st.text_area('Text to analyze', )

# st.write('This text is:', len(txt))


# if st.button('click me'):
#     # print is visible in the server output, not in the page
#     print('button clicked!')
#     st.write('I was clicked 🎉')
#     st.write('Further clicks are not visible but are executed')
# else:
#     st.write('I was not clicked 😞')
