from transformers import DistilBertTokenizer, DistilBertConfig, TFDistilBertForSequenceClassification
import tensorflow as tf
import streamlit as st
import requests
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords
import nltk

# ------------------------------------------------------------------------------------------------#

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

url_model = 'https://storage.googleapis.com/fake-news-detection-wagon/distilbert_model_best.h5/tf_model.h5'
url_config = 'https://storage.googleapis.com/fake-news-detection-wagon/distilbert_model_best.h5/config.json'

def download_file(url, output_path):
    response = requests.get(url)
    with open(output_path, "wb") as file:
        file.write(response.content)

output_path_model = './tf_model.h5'
output_path_config = './config.json'

download_file(url_model, output_path_model)
download_file(url_config, output_path_config)

config_model = DistilBertConfig.from_json_file('config.json')

model = TFDistilBertForSequenceClassification.from_pretrained('tf_model.h5', config=config_model)

# ------------------------------------------------------------------------------------------------#

# Download the dictionary for stopwords
nltk.download('stopwords')

# Get the set of stopwords
stop_words_set = set(stopwords.words('english'))

# Load English tokenizer from spacy
nlp = English()
spacy_tokenizer = nlp.tokenizer ## make instance

# Create function to clean text -- lowercase, remove non alphanumeric, remove stop words
def optimized_preprocess(text): ## Takes in a list of texts, i.e. the entire corpus
    # Tokenize using spaCy’s tokenizer
    tokens = [token.text.lower() for token in spacy_tokenizer(text) if token.text.isalpha() and token.text.lower() not in stop_words_set]
    cleaned_query= ' '.join(word for word in tokens)
    return cleaned_query

# ------------------------------------------------------------------------------------------------#

threshold = 0.7

def test_article(article, optimizer=True, max_length=300):
    if optimizer == True:
    # store preprocessed text in random_input variable
        random_input = optimized_preprocess(article)
    else:
        random_input = article
    # tokenize the random input and return as a tensor
    random_input = tokenizer.encode_plus(
        random_input,
        add_special_tokens=True,
        max_length=max_length,
        truncation = True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors="tf"
    )
    # Extract 'input_ids' from the 'random_input' dictionary.
    # (numerical representations of textual input for the model)
    input_ids = random_input['input_ids']
    # Extract the 'attention_mask' from the 'random_input' dictionary.
    # This mask helps the model focus on relevant parts of the input.
    attention_mask = random_input['attention_mask']
    # Pass the 'input_ids' and 'attention_mask' to the model to get predictions.
    # The model uses these inputs to make predictions about the class of the input (Real or Fake).
    predictions = model(input_ids, attention_mask=attention_mask)
    # Apply the softmax function to the logits (raw outputs) of the model's predictions.
    # Softmax converts logits to probabilities, making them easier to interpret.
    probabilities = tf.nn.softmax(predictions.logits, axis=-1)
    max_probability = tf.reduce_max(probabilities, axis=-1).numpy()[0]

    if max_probability < threshold:
        classification = "**Suspicious**"
    else:
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()
        class_names = ['Real', 'Fake']
        classification = f"**{class_names[predicted_class[0]]}**"

    return [f"The article is predicted as: {classification}", f"Probabilities per class: {probabilities.numpy()[0]}"]

# ------------------------------------------------------------------------------------------------#

# Streamlit app
st.title("CredibleContent 📰")
st.markdown('''
            The technological surge in the past few years has led to a plethora of
            misinformation being spread among the vast corners of the Internet.
            ''')
st.markdown('''
            This is a news detector that aims to predict whether a given text, namely a news article,
            conveys real information, fake information, or is rather suspicious.
            ''')
st.markdown('''
            All you need to do is input a text below and it will return a prediction,
            as well as its probability.
            ''')

# User input for prediction
user_input = st.text_area("Text input", )

if st.button("Predict"):
    # Make prediction when the button is clicked
    if user_input.strip() != "":
        prediction_result = test_article(user_input, optimizer=True, max_length=500)
        # st.write(f"**{prediction_result[0]}**")
        st.write(prediction_result[0])
        if "Suspicious" in prediction_result[0]:
            st.warning('Hmm... this article does not seem credible. It might be best to do further research on its contents. 🧐')
        if "Fake" in prediction_result[0]:
            st.error('Hmm... this article is very likely to be providing false information. 🥸 We advise to use judgement and conduct further research on its contents.')
        if "Real" in prediction_result[0]:
            st.success('This article contains credible information! 😎')
        st.write(prediction_result[1])
    else:
        st.warning("Please enter text for prediction.")
