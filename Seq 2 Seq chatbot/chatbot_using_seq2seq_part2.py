import numpy as np
import tensorflow as tf
import pickle
#import keras
from tensorflow.keras import layers , activations , models , preprocessing
from keras.models import load_model

def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( word_dict[ word ] )
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_question_len , padding='post')


encoder_input_data = np.load( 'chatbot_nlp_processed/data/enc_in_data.npy' )
decoder_input_data = np.load( 'chatbot_nlp_processed/data/dec_in_data.npy' )
decoder_target_data = np.load( 'chatbot_nlp_processed/data/dec_tar_data.npy' )
embedding_matrix = np.load('chatbot_nlp_processed/data/embedding.npy' ) 
tokenizer = pickle.load( open('chatbot_nlp_processed/data/tokenizer.pkl' , 'rb'))

num_tokens = len( tokenizer.word_index )+1
word_dict = tokenizer.word_index

max_question_len = encoder_input_data.shape[1]
max_answer_len = decoder_input_data.shape[1]

encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True , weights=[embedding_matrix] ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True, weights=[embedding_matrix]) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')
model =  tf.keras.models.load_model('model.h5');
enc_model, dec_model = make_inference_models()


def response(input_sentence):
    states_values = enc_model.predict( str_to_tokens( input_sentence ) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in word_dict.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > max_answer_len:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    return decoded_translation

while(True):
    user_msg=input('->')
    reply=response(user_msg)
    print(reply)
