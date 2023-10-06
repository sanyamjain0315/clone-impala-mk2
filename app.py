import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



def generate_lyrics(seed_text, next_words):
    # Loading the tokenizer
    with open('models\\YOUR_TOKENIZER,pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    max_sequence_len = tokenizer.num_words

    # Loading the model
    model = load_model('models\\YOUR_MODEL.h5')
    max_sequence_len = model.input_shape[1] + 1

    # Generting lyrics
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], 
                                   maxlen=max_sequence_len-1, 
                                   padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text



if __name__=="__main__":
    page_bg_img = '''
    <style>
        .stApp {
    background-image: url("https://wallpapercave.com/wp/wp5576823.jpg");
    background-size: cover;
    }
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title("Clone impala")
    seed_text = st.text_input("Enter first few words of your song")
    next_words = st.number_input("Number of words to generate",0,250,value=100)
    if st.button("Generate Lyrics"):
        with st.spinner(text="Generating..."):
            generated_lyrics = generate_lyrics(seed_text, next_words)
            st.header("Lyrics")
            st.markdown(f'<div style="background-color:#262730; padding:10px; border-radius:5px;">{generated_lyrics}</div>', unsafe_allow_html=True)