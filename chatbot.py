import json
import string
import random
import nltk # you need to: pip install nltk
import numpy as np # you nedde to: pip install numpy
from nltk.stem import WordNetLemmatizer
import tensorflow as tf # you need to: pip install tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from data import *

nltk.download("punkt")

"""
WordNet é um banco de dados publico de mais de 200 idiomas que provê
relações semânticas (sinônimos) entre as palavras. Esta presente na 
lib nltk do python. Wordnet agrupa os sinônimos em form de 'synsets'.
    - synsets: um grupo de dados de elementos que são semanticamente relacionados
"""
# Utiliza a nltk para baixar o WordNet
nltk.download("wordnet")

"""
O lemmatizer é utilizado para obter a raiz das palavras. 
(!) Usar sempre minucsculas.
Exemplo:
    Original -----------> Raiz (lemmatizado)
      andando  ---------> andar
      abelhas ----------> abelha
      era     ----------> é
"""
# Inicia o lemmatizer 
lemmatizer = WordNetLemmatizer()

"""
Afim de organizar os dados e separa-los, criam-se 3 listas:
- Um vocabulario com todas as palavras utilizadas nas perguntas do usuario
- Uma lista de classes (as tags de cada intention)
- Uma lista com todas as perguntas
- Uma lista com todas as tags (assuntos) associadas a cada pattern(pergunta) no arquivo de intentions
"""
# Cria as listas
words = []
classes = []
doc_x = []
doc_y = []

# Usa um loop for para percorrer todas as intentions
# cria um token para cada pattern (pergunta) e inclui o token na lista words
# os patterns e as tags são adicionadas em suas listas respectivas
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])

    # Adiciona a tag às classes se ainda não estiver
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemmatizar todas as palavras do vocabulario e converter em ninuscula e ignorar a pontuação
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# Organizar o vocabulario por ordem alfabética 
# set() para assegurar que não vai haver duplicadas
words = sorted(set(words))
classes = sorted(set(classes))

""" 
TRATAMENTO DOS DADOS 
        As redes neurais esperam receber numeros 
        ao invés de palavras. Faz-se então o tratamento
        dos dados para que a rede neural possa lê-los
"""
training = [] # Lista para os dados de treinamento
out_empty = [0] * len(classes)

# Cria um modelo de conjunto de palavras
for idx, doc in enumerate(doc_x):
    bow = [] # BoW bag of words
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # Marca o index da classe à qual o pattern(pergunta) é associado
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    # Adiciona a BoW e as classes associadas à lista de treinamento
    training.append([bow, output_row])

# Embaralha os dados e converte num array
random.shuffle(training)
training = np.array(training, dtype=object)

# Separa as caracteristicas 
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

""" Agora os dados ja estão convertidos em array (formato numerioc)"""


""" 
        REDE NEURAL
"""
# Definir os paramentros
input_shape = (len(train_x[0]),)
output_shape = len(train_y[0])
epochs = 200

# Deep Learnig
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation="softmax"))

adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

#print(model.summary())

# Treinando o modelo
model.fit(x=train_x, y=train_y, epochs=200, verbose=1)

"""
Funções do chatbot
"""

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


# Chatbot V1.0
while True:
    message = input("#")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)

