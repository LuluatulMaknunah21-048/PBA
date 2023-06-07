import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import string
from sklearn.pipeline import Pipeline
import re
import scipy.sparse as sp

with st.sidebar:
    selected=option_menu('Analisis Sentimen',['Tentang Kami','Start Analisis'])
if selected=='Tentang Kami':
    st.title('WEB ANALISIS SENTIMEN')
    st.write('Selamat datang,')
    st.write('website ini merupakan website untuk melakukan analisis sentimen dari kalimat yang di inputkan oleh pengguna, website ini menggunakan metode KNN untuk melakukan analisis sentimen dengan cosine similarity dan tf idf sehingga akurasi dari prediksi yang dilakukan memiliki akurasi yang bagus.') 
    
if selected=='Start Analisis':
    st.title('ANALISIS SENTIMEN')
    text = st.text_input("Masukkan teks").lower()
    button=st.button('START ANALISIS')
    if button :
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        import re

        # Menginisialisasi Streamlit
        #st.title("Preprocessing pada Teks")

        # Mengaktifkan resource NLTK yang diperlukan
        nltk.download('punkt')
        nltk.download('stopwords')
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        
        # Membaca kamus dari file Excel
        kamus_df = pd.read_excel('PK')
        # Mengubah kamus menjadi dictionary
        kamus_dict = dict(zip(kamus_df['before'], kamus_df['after']))
        def normalize_typo(text):
            words = text.split()
            normalized_words = []
            for word in words:
                if word in kamus_dict:
                    corrected_word = kamus_dict[word]
                    normalized_words.append(corrected_word)
                else:
                    normalized_words.append(word)
            normalized_text = ' '.join(normalized_words)
            return normalized_text
        
        # Mendefinisikan fungsi pra-pemrosesan
        def preprocess_text(text):
            # Menghilangkan karakter yang tidak diinginkan
            text = text.strip(" ")
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'[?|$|.|!_:")(-+,]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            text = re.sub(r"\b[a-zA-Z]\b", " ",text)
            text = re.sub('\s+',' ', text)
            text = normalize_typo(text)
            # Tokenisasi teks menjadi kata-kata
            tokens = word_tokenize(text)
            
            # Menghapus kata-kata yang tidak bermakna (stopwords)
            stop_words = set(stopwords.words('Indonesian'))
            tokens = [token for token in tokens if token not in stop_words]
            
            # Menggabungkan kata-kata kembali menjadi teks yang telah dipreprocessed
            processed_text = ' '.join(tokens)
            
            # Melakukan stemming pada teks menggunakan PySastrawi
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stemmed_text = stemmer.stem(processed_text)
            return stemmed_text

# Mengambil input teks dari pengguna
        #st.write("Hasil Preprocessing:")
        analisis=preprocess_text(text)
        #st.write(analisis)
        
        import pickle
        with open ('tanpacosin.pkl', 'rb') as r:
            asknn=pickle.load(r)
        import pickle
        with open('tfidf.pkl', 'rb') as f:
            vectoriz= pickle.load(f)    
        
        
        hastfidf=vectoriz.transform([analisis])
        predictions = asknn.predict(hastfidf)
        for i in predictions:
            st.write('Text : ',analisis)
            st.write('Sentimen :', i)
        #Menampilkan hasil prediksi
        #sentiment = asknn.predict(cosim)
        #st.write("Sentimen:", sentiment)


    
