import keras.initializers.initializers_v2
from keras.layers import Input, Embedding, Concatenate, TimeDistributed, Dense
from keras.models import Model
from keras import initializers

from models.lstm_stack import build_bilstm_stack


def build_model(n_layers,
                hidden_size, 
                input_length,
                words_params = None,
                tags_params = None,
                deps_params = None,
                dropout_rate=0.2):

    (words_size, words_emb_size, embedding_matrix) = words_params if words_params else (0, 0, None)
    (tags_size, tags_emb_size) = tags_params if tags_params else (0, 0)
    (deps_size, deps_emb_size) = deps_params if deps_params else (0, 0)
    
    inputs_size = words_emb_size + \
                  tags_emb_size + \
                  deps_emb_size
                 
    outputs_size = 2
    
    inputs = []
    inputs_emb = []

    if words_params:
        words_inputs = Input(shape=(input_length,), name='words_inputs')
        words_emb_layer = Embedding(words_size, words_emb_size, embeddings_initializer=keras.initializers.initializers_v2.Constant(embedding_matrix), mask_zero=True, name='words_inputs_emb')
        words_inputs_emb = words_emb_layer(words_inputs)
        
        inputs.append(words_inputs)
        inputs_emb.append(words_inputs_emb)
    
    if tags_params:
        tags_inputs = Input(shape=(input_length,), name='tags_inputs')
        tags_emb_layer = Embedding(tags_size, tags_emb_size, mask_zero=True, name='tags_inputs_emb')
        tags_inputs_emb = tags_emb_layer(tags_inputs)
        
        inputs.append(tags_inputs)
        inputs_emb.append(tags_inputs_emb)
    
    if deps_params:
        deps_inputs = Input(shape=(input_length,), name='deps_inputs')
        deps_emb_layer = Embedding(deps_size, deps_emb_size, mask_zero=True, name='deps_inputs_emb')
        deps_inputs_emb = deps_emb_layer(deps_inputs)
        
        inputs.append(deps_inputs)
        inputs_emb.append(deps_inputs_emb)

    concat = Concatenate(name='concat')
    bilstm = build_bilstm_stack(n_layers, hidden_size, inputs_size, input_length, dropout_rate)
    classifier = TimeDistributed(Dense(outputs_size, activation='softmax', name='classifier'))

    flow = concat(inputs_emb) if len(inputs_emb) > 1 else inputs_emb[0]
    hidden_states = bilstm(flow)
    outputs = classifier(hidden_states)

    return Model(inputs=inputs, outputs=outputs)