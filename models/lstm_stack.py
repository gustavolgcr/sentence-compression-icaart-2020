from tensorflow.keras.layers import Input, LSTM, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K


def build_lstm_stack(input_size, hidden_units, dropout_rate=-1, return_sequences=False, go_backwards=False, name='lstm_stack', print_summary=False):
    def interleave(main_list, interleave):
        size = len(main_list) + len(interleave)

        return list(map(lambda i: main_list[i//2] if i % 2 == 0 else interleave[i//2], range(size)))
    
    go_backwards_fn = lambda i: go_backwards and i == 0
    return_sequences_fn = lambda i: not i == (len(hidden_units) - 1) or return_sequences
    
    inputs = Input(shape=(None, input_size,), name='inputs')
    flow = inputs
    
    stack = [
        LSTM(hidden_unit, 
             return_sequences=return_sequences_fn(l), 
             go_backwards=go_backwards_fn(l), 
             name='rnn_layer_%d' % l)
        for l, hidden_unit in enumerate(hidden_units)
    ]
    
    if dropout_rate > 0:
        dropout_layers = [Dropout(rate=dropout_rate) for _ in hidden_units[:-1]]
        stack = interleave(stack, dropout_layers)
    
    for rnn in stack:
        flow = rnn(flow)
    
    outputs = flow
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    if print_summary: 
        model.summary()
        
    return model


def build_bilstm_stack(n_layers, hidden_size, input_size, input_length, dropout_rate, return_sequences=True, name='bilstm_stack'):
    inputs = Input(shape=(input_length, input_size), name='classifier_inputs')

    forward = build_lstm_stack(input_size, 
                               [hidden_size] * n_layers, 
                               dropout_rate=dropout_rate, 
                               return_sequences=return_sequences, 
                               name='forward')
    
    backward = build_lstm_stack(input_size, 
                                [hidden_size] * n_layers, 
                                dropout_rate=dropout_rate, 
                                return_sequences=return_sequences, 
                                go_backwards=True,
                                name='backward')
    concat = Concatenate(name='concat')

    outputs_forward = forward(inputs)
    outputs_backward = Lambda(lambda x: K.reverse(x, axes=[1]))(backward(inputs))

    outputs = concat([outputs_forward, outputs_backward])

    return Model(inputs=inputs, outputs=outputs, name=name)