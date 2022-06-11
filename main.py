import random

import numpy as np
import tensorflow as tf

from preprocessing import Preprocessor

from utils.utils import save_obj, load_obj, split_train_dev_test
from models.bilstm import build_model

from keras.callbacks import ModelCheckpoint

data = "data"


def get_new_dataset(number_of_sentences):

    words_inputs = np.load('%s/inputs/replaced_words_inputs_augmented.npy' % data)
    tags_inputs = np.load('%s/inputs/tags_inputs_augmented.npy' % data)
    deps_inputs = np.load('%s/inputs/deps_inputs_augmented.npy' % data)
    outputs = np.load('%s/inputs/outputs_augmented.npy' % data)

    sentences_index_list = random.sample( range(0, int(len(words_inputs))), number_of_sentences + 5 )

    words_inputs = np.take(words_inputs, sentences_index_list, axis=0)
    tags_inputs = np.take(tags_inputs, sentences_index_list, axis=0)
    deps_inputs = np.take(deps_inputs, sentences_index_list, axis=0)
    outputs = np.take(outputs, sentences_index_list, axis=0)

    return words_inputs, tags_inputs, deps_inputs, outputs


def train_model():
    trial = 'bilstm'

    max_input_length = 130

    n_layers = 3
    hidden_size = 100

    words_emb_size = 100
    tags_emb_size = 40
    deps_emb_size = 40

    dropout_rate = 0.5
    learning_rate = 0.001

    batch_size = 30
    n_epochs = 1

    train_size = 2000

    number_of_pairs = [80, 160, 320, 640, 1280, 2560]

    model_name = '%s_%dL_%dH_%dEW_%dET_%dED_%dk_t2' % (trial, n_layers, hidden_size, words_emb_size, tags_emb_size,
                                                       deps_emb_size, (train_size // 1000))

    words_vocab = load_obj('%s/vocabs/replaced_words_vocab_augmented' % data)
    tags_vocab = load_obj('%s/vocabs/tags_vocab_augmented' % data)
    deps_vocab = load_obj('%s/vocabs/deps_vocab_augmented' % data)

    words_vocab_size = len(words_vocab)
    tags_vocab_size = len(tags_vocab)
    deps_vocab_size = len(deps_vocab)

    words_embedding_matrix = np.load('%s/embeddings/replaced_words_glove_matrix_augmented.npy' % data)

    words_inputs = np.load('%s/inputs/replaced_words_inputs_augmented.npy' % data)
    tags_inputs = np.load('%s/inputs/tags_inputs_augmented.npy' % data)
    deps_inputs = np.load('%s/inputs/deps_inputs_augmented.npy' % data)
    outputs = np.load('%s/inputs/outputs_augmented.npy' % data)

    number_of_sentences = 400
    words_inputs, tags_inputs, deps_inputs, outputs = get_new_dataset(number_of_sentences)

    words_inputs_train, words_inputs_dev, _ = split_train_dev_test(words_inputs, number_of_sentences, 5)
    tags_inputs_train, tags_inputs_dev, _ = split_train_dev_test(tags_inputs, number_of_sentences, 5)
    deps_inputs_train, deps_inputs_dev, _ = split_train_dev_test(deps_inputs, number_of_sentences, 5)
    outputs_train, outputs_dev, _ = split_train_dev_test(outputs, number_of_sentences, 5)

    models_path = 'models'
    logs_path = 'logs'

    for i in range(0, 1):
        print('Training model %d ...' % i)
        model = build_model(
            n_layers,
            hidden_size,
            max_input_length,
            (words_vocab_size, words_emb_size, words_embedding_matrix),
            (tags_vocab_size, tags_emb_size),
            (deps_vocab_size, deps_emb_size),
            dropout_rate=dropout_rate
        )

        model.compile('adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        checkpoint_callback = ModelCheckpoint('%s/%s/%s_%d.h5' % (models_path, trial, model_name, i),
                                              monitor='val_categorical_accuracy', save_best_only=True)

        training = model.fit(
            [words_inputs_train[:train_size, :],
             tags_inputs_train[:train_size, :],
             deps_inputs_train[:train_size, :]], outputs_train[:train_size, :, :],
            validation_data=[[words_inputs_dev, tags_inputs_dev, deps_inputs_dev], outputs_dev],
            batch_size=batch_size,
            epochs=n_epochs,
            initial_epoch=0,
            verbose=1,
            callbacks=[
                checkpoint_callback
            ],
        )

        save_obj(training.history, '%s/%s/%s_%d_history' % (logs_path, trial, model_name, i))


def main():

    sentences_file_path = data + "/google/sentence_en.txt"
    compressions_file_path = data + "/google/compression_en.txt"

    preprocessor = Preprocessor(sentences_file_path, compressions_file_path)
    preprocessor.preprocess_data()

    train_model()


if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
