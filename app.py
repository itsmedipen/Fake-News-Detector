
import streamlit as st
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
import string
import nltk
from nltk.tokenize import RegexpTokenizer
def transform_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text.lower())  # use regex tokenizer only

    y = []

    for i in text:  # removng special characters only selecting alpha numeric
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    stop_words = ENGLISH_STOP_WORDS
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(snow.stem(i))
        
    return ' '.join(y)



# Force load the standard punkt

snow = SnowballStemmer('english')

# Load the model and vectorizer
with open('Model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Tfidf.pkl','rb') as f:
    vector = pickle.load(f)

st.title("Fake news classification")

# User input
user_input = st.text_area('Enter your message:')

if st.button("Predict",type= 'primary') and user_input.strip() != '':
    with st.spinner('Predicting...'):
        #pass to the transform text

        transform_sms = transform_text(user_input)
        # Transform the user input
        vector_input = vector.transform([transform_sms]).toarray()

        
        # Make prediction
        prediction = model.predict(vector_input)
        
        # Show result
        if prediction[0] == 0:
            st.error("⚠️ Alert: This is a fake news")
        else:
            st.success("✅ This is a real news")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 0.9em;
        color: gray;
    }
    </style>
    <div class="footer">
        🧑Developed by <strong>Dipen Sherpa</strong>&nbsp;|&nbsp; © 2025
    </div>
    """,
    unsafe_allow_html=True
)
