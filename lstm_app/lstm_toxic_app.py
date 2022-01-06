import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Toxic Comment Classifier")
st.text("Provide text for comment classification")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('C:/Users/lstm_app/models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

#classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000

def decode_text(text,MAX_VOCAB_SIZE):
  tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
  tokenizer.fit_on_texts([text])
  text = tokenizer.texts_to_sequences([text])
  text = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)
  return text



text = st.text_input('Enter text to Classify.. ',"Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info")
if text is not None:
    #content = requests.get(path).content

    st.write("Predicted Probabilities:")

    with st.spinner('classifying.....'):
      proba = model.predict(decode_text(text,MAX_VOCAB_SIZE))[0]
      data = {'Labels':possible_labels,'Probability':proba}
      df = pd.DataFrame(data)
      st.write(df)    
    st.write("")
    #image = Image.open(BytesIO(content))
    #st.image(image, caption='Classifying Image', use_column_width=True)


    ##MEL GIBSON IS A NAZI BITCH WHO MAKES SHITTY MOVIES. HE HAS SO MUCH BUTTSEX THAT HIS ASSHOLE IS NOW BIG ENOUGH TO BE CONSIDERED A COUNTRY.