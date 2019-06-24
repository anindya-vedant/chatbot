import json
import nltk


words=[]
classes=[]
documents=[]
ignore_words=["?"]  #to ignore question marks



with open('intents.json') as json_data:  #loading the training data from json
    intents=json.load(json_data)
for intent in intents['intents']:                                       #
    for pattern in intent['patterns']:
        print("pattern is ",pattern)                                                  #
        word=nltk.word_tokenize(pattern)
        print("word is ",word)                                                     #
        words.extend(word)
        print("tag is ",intent['tag'])#preparing training data, that is sorted into words and classes
        documents.append((word,intent['tag']))                          #tokenizing the words
        if intent['tag'] not in classes:                                #
            classes.append(intent['tag'])  
