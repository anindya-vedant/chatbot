import pickle
import json
import random
import tensorflow as tf
import numpy as np
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer


def clean_up_sentence(sentence):
    #this function is used to clean up a sentence, which means that the words are tokenized, converted to lower case, stemmed and then returned
    # sentence=sentence.strip()
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence,words,show_details=False):

    #first of the all the sentence is cleaned up and a "bag of words" is constructed according to the input sentence and the numpy array of the bag
    #of words is returned.
    sentence_words=clean_up_sentence(sentence)

    bag=[0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1
                if show_details:
                    print("found in bag: %s" %w)

    return(np.array(bag))

def classify(sentence):

    #use the pre-trained model to predict the class/sentiment of the input class
    #to decide the most appropriate response for the input.
    results=model.predict([bow(sentence,words)])[0]
    results=[[index,literature_word_vector] for index, literature_word_vector in enumerate(results) if literature_word_vector>error_threshold]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append((classes[r[0]],r[1]))
    return return_list

def response(sentence,userID='123',show_details=False):

    #use the result gained from "classify" function to randomly choose a response from a list of responses

    results=classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag']==results[0][0]:
                    return random.choice(i['responses'])

            results.pop(0)


stemmer=LancasterStemmer()                                                          #
data = pickle.load(open("training_data","rb"))                                      #
words=data['words']                                                                 #intializing a stemmer object and loading necessary info required
classes=data['classes']                                                             #like words, classes, training data of words (x) and labels (y)
train_x=data['train_x']                                                             #
train_y=data['train_y']                                                             #

with open('intents.json') as json_data:                                             #loaded for selecting responses.
    intents = json.load(json_data)

neural_network=tflearn.input_data(shape=[None,len(train_x[0])])                                         #
neural_network = tflearn.fully_connected(neural_network,8)                                              #
neural_network = tflearn.fully_connected(neural_network,8)                                              #declaring the same neural network as the original network
neural_network = tflearn.fully_connected(neural_network,len(train_y[0]),activation='softmax')           #used for training
neural_network = tflearn.regression(neural_network)                                                     #

model=tflearn.DNN(neural_network,tensorboard_dir='tflearn_logs')

model.load('./model.tflearn')                                                       #loading the pre-trained model

error_threshold=0.25                                                                #thresholding the loss function.

# print("Hello! I am Skynet.")
#
# while(1):
#     msg=input("-> ")
#     if (msg in ["quit","bye","goodbye","exit"]):
#         break
#     else:
#         response(msg)


