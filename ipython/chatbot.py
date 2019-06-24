import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,CuDNNLSTM
import json
import pickle


with open('intents.json') as json_data:  #loading the training data from json
    intents=json.load(json_data)
stemmer = LancasterStemmer()    #intiliazing a stemmer object

words=[]
classes=[]
documents=[]
ignore_words=["?"]  #to ignore question marks


for intent in intents['intents']:                                       #
    for pattern in intent['patterns']:                                  #
        word=nltk.word_tokenize(pattern)                                #
        words.extend(word)                                              #preparing training data, that is sorted into words and classes
        documents.append((word,intent['tag']))                          #tokenizing the words
        if intent['tag'] not in classes:                                #
            classes.append(intent['tag'])                               #


words=[stemmer.stem(word.lower()) for word in words if word not in ignore_words]
words=sorted(list(set(words)))   #stemmed words are fetched in lowercase and then sorted alphabetically


classes=sorted(list(set(classes)))

print(len(documents),"documents")
print(len(classes),"classes", classes)
print(len(words),"unique stemmed words",words)

training=[]
output=[]

output_empty=[0]*len(classes) #output label array


for doc in documents:
    bag=[]
    pattern_words=doc[0]
    pattern_words=[stemmer.stem(w.lower()) for w in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0) #denoting membership of the word using 1

    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1

    training.append([bag,output_row]) #preparing training data

random.shuffle(training)
training=np.array(training)


train_x=list(training[:,0]) #preparing training data of words
train_y=list(training[:,1]) #preparing training data of labels


tf.reset_default_graph()                                                                                #

neural_network=tflearn.input_data(shape=[None,len(train_x[0])])                                         # shaping the first input layer of the neural network
neural_network = tflearn.fully_connected(neural_network,8)                                              # choosing the second layer of the neural network
neural_network = tflearn.fully_connected(neural_network,8)                                              #
neural_network = tflearn.fully_connected(neural_network,len(train_y[0]),activation='softmax')           # second last layer has the number of nodes equal to the number of the classes and activation function as softmax
neural_network = tflearn.regression(neural_network)                                                     # the regression is applied to the output to further judge the accuracy of the output

model=tflearn.DNN(neural_network,tensorboard_dir='tflearn_logs')                                        #constructing a Deep Neural Network and logging it into tensorboard for visualization purposes

model.fit(train_x,train_y,n_epoch=5000,batch_size=8,show_metric=True)                                   #start training of the model
model.save('model.tflearn')                                                                             #save the trained model

pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )  #make a json dump of the useful data in an organized manner
