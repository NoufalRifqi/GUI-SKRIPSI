import pandas as pd
import numpy as np
import ast
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


class Helper(object):

    def preprocessing(self, text):
        #Cleaning
        text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) # menghapus http/https (link)
        text = re.sub(r'[^\w\s]', ' ', text) # menghilangkan tanda baca
        text = re.sub('<.*?>', ' ', text) # mengganti karakter html dengan tanda petik
        text = re.sub('[\s]+', ' ', text) # menghapus spasi berlebihan
        text = re.sub('[^a-zA-Z]', ' ', text) # mempertimbangkan huruf dan angka
        text = re.sub("\n", " ", text) # mengganti line baru dengan spasi
        text = ' '.join(text.split()) # memisahkan dan menggabungkan kata
         
        #Case Folding
        text = text.lower() # mengubah ke huruf kecil
        
        #Tokenize
        regexp = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
        text = regexp.tokenize(text)
    
        #Stopword
        list_stopword = set(stopwords.words('indonesian'))
        hapus_kata = {"tidak", "soal", "ada", "belum", "lama", "jawaban", "kelamaan", "kurang", "jawab"}
        list_stopword.difference_update(hapus_kata)
        text = [token for token in text if token not in list_stopword]
        
        #Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        text = [stemmer.stem(word) for word in text]

        text = [' '.join(text)]

        return text

    def text2seqpad(self, text):
        dataset = pd.read_csv("data/Hasil Processing Imbalance.csv")
        dataset["processing_result"] = dataset["processing_result"].apply(lambda x: ast.literal_eval(x))
        X = dataset["processing_result"].tolist()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        max_length = max([len(s) for s in sequences])
        sequence = tokenizer.texts_to_sequences(text)
        padding = pad_sequences(sequence, maxlen=max_length)
        return padding

    def model_classification(self, input):
        model = load_model("model/model_1dcnn")
        predict = model.predict(input)
        labels = ["NEGATIF", "NETRAL", "POSITIF"]
        classification = labels[np.argmax(predict)]
        return classification