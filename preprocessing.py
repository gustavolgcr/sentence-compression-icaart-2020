import numpy as np
import pandas as pd

import spacy
from spacy.tokens import Doc

from nltk.tokenize import word_tokenize
from utils.utils import save_obj, load_obj

from collections import Counter


class Preprocessor:
    def __init__(self, sentences_file_path, compressions_file_path):
        self.data = "data"
        self.sentences_file_path = sentences_file_path
        self.compressions_file_path = compressions_file_path
        self.sentences = []
        self.compressions = []
        self.sentences_tokenized = []
        self.compressions_tokenized = []
        self.dictionary_compression = []

        self.substitute_words = {
            'PERSON': 'name',
            'NORP': 'name',
            'FAC': 'name',
            'ORG': 'name',
            'GPE': 'name',
            'LOC': 'name',
            'PRODUCT': 'name',
            'EVENT': 'name',
            'WORK_OF_ART': 'name',
            'LAW': 'law',
            'LANGUAGE': 'language',
            'DATE': 'date',
            'TIME': 'moment',
            'PERCENT': 'number',
            'MONEY': 'number',
            'QUANTITY': 'measurement',
            'ORDINAL': 'number',
            'CARDINAL': 'number'
        }

    def preprocess_data(self):
        self.__data_import()
        self.__tokenizer()
        self.__dictionary_creation()
        self.__feature_extraction()
        self.__word_replacement()
        self.__vectorization()
        self.__build_glove_matrix()

    def __data_import(self):
        file_sentences = open(self.sentences_file_path, "r")
        file_compressions = open(self.compressions_file_path, "r")

        for line in file_sentences:
            self.sentences.append(line[:-1])

        for line in file_compressions:
            self.compressions.append(line[:-1])

        return self.sentences, self.compressions

    def __tokenizer(self):
        for sentence in self.sentences:
            sentence_tokenized = []

            for token in word_tokenize(sentence):
                sentence_tokenized.append(token)

            self.sentences_tokenized.append(sentence_tokenized)

        for compression in self.compressions:
            compression_tokenized = []

            for token in word_tokenize(compression):
                compression_tokenized.append(token)

            self.compressions_tokenized.append(compression_tokenized)

        return self.sentences_tokenized, self.compressions_tokenized

    def __dictionary_creation(self):
        sentences_test = self.sentences_tokenized.copy()
        compressions_test = self.compressions_tokenized.copy()

        list_result = []

        for i, sentence in enumerate(sentences_test):
            list_result_temp = []
            dict_result = {}

            for j, sentence_word in enumerate(sentence):
                breaking_flag = 0
                for k, compression_word in enumerate(compressions_test[i]):
                    if sentence_word == compressions_test[i][k]:
                        list_result_temp.append(1)
                        del compressions_test[i][k]
                        breaking_flag = 1
                        break

                if breaking_flag == 1:
                    continue

                list_result_temp.append(0)
            dict_result['labels'] = list_result_temp
            dict_result['sentence'] = ' '.join(sentence)
            self.dictionary_compression.append(dict_result)
            list_result.append(list_result_temp)

        datasets_file_path = '%s/support_files/sentences_augmented' % self.data
        save_obj(self.dictionary_compression, datasets_file_path)

        return self.dictionary_compression

    def __feature_extraction(self):
        nlp = spacy.load('en_core_web_sm', disable=["textcat", "sentencizer"])

        datasets_file_path = '%s/support_files/sentences_augmented' % self.data
        sentences = load_obj(datasets_file_path)

        total_steps = len(sentences)

        nlp.tokenizer = SeparatorTokenizer(nlp.vocab)

        sentences_docs = [{'sentence': self.__log(step, nlp, total_steps)(entry['sentence']), 'labels': entry['labels']}
                          for step, entry in enumerate(sentences)]

        sentences_feats = [self.__extract_features_from_sentence(sentence['sentence'], sentence['labels'])
                           for sentence in sentences_docs]

        datasets_file_path = '%s/support_files/sentences_feats_augmented' % self.data
        save_obj(sentences_feats, datasets_file_path)

    def __word_replacement(self):

        datasets_file_path = '%s/support_files/sentences_feats_augmented' % self.data

        sentences_feats = load_obj(datasets_file_path)

        sentences_replaced = [self.__replace_words(sentence) for sentence in sentences_feats]

        datasets_file_path = '%s/support_files/sentences_replaced_augmented' % self.data

        save_obj(sentences_replaced, datasets_file_path)

    def __vectorization(self):
        datasets_file_path = '%s/support_files/sentences_replaced_augmented' % self.data
        sentences_replaced = load_obj(datasets_file_path)

        words_with_frequencies = Counter([word['original'] for sentence in sentences_replaced for word in sentence])
        replaced_words_with_frequencies = Counter([word['text'] for sentence in sentences_replaced for word in sentence])

        max_occurrences = 3

        words_vocab = ['', '<UNK>'] + \
                      sorted([word for word, occurrences in words_with_frequencies.items() if
                              occurrences >= max_occurrences])

        replaced_words_vocab = ['', '<UNK>'] + \
                               sorted([word for word, occurrences in replaced_words_with_frequencies.items() if
                                       occurrences >= max_occurrences])

        tags_vocab = ['', '<UNK>', '-LRB-', '-RRB-', ',', ':', '.', "'", '""', '``', '#', '$', 'ADD', 'AFX', 'BES',
                      'CC',
                      'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NIL',
                      'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', '_SP', 'SYM',
                      'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX']

        deps_vocab = ['', '<UNK>', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux',
                      'auxpass',
                      'case', 'cc', 'ccomp', 'compound', 'conj', 'cop', 'csubj', 'csubjpass', 'dative', 'dep', 'det',
                      'dobj',
                      'expl', 'intj', 'mark', 'meta', 'neg', 'nn', 'nounmod', 'npmod', 'nsubj', 'nsubjpass', 'nummod',
                      'oprd',
                      'obj', 'obl', 'parataxis', 'pcomp', 'pobj', 'poss', 'preconj', 'prep', 'prt', 'punct', 'quantmod',
                      'relcl', 'root', 'xcomp', ]

        save_obj(words_vocab, '%s/vocabs/words_vocab_augmented' % self.data)
        save_obj(replaced_words_vocab, '%s/vocabs/replaced_words_vocab_augmented' % self.data)
        save_obj(tags_vocab, '%s/vocabs/tags_vocab_augmented' % self.data)
        save_obj(deps_vocab, '%s/vocabs/deps_vocab_augmented' % self.data)

        max_length = 130

        sentences_replaced_filtered = [sentence for sentence in sentences_replaced if len(sentence) <= max_length]

        save_obj(sentences_replaced_filtered, '%s/sentences_replaced_filtered_augmented' % self.data)

        word2index = dict([(word, index) for index, word in enumerate(words_vocab)])
        replaced_word2index = dict([(word, index) for index, word in enumerate(replaced_words_vocab)])
        tags2index = dict([(tag, index) for index, tag in enumerate(tags_vocab)])
        deps2index = dict([(dep, index) for index, dep in enumerate(deps_vocab)])

        words_seq = [[self.__get_index(word['original'], word2index) for word in sentence] for sentence in
                     sentences_replaced_filtered]
        replaced_words_seq = [[self.__get_index(word['text'], replaced_word2index) for word in sentence] for sentence in
                              sentences_replaced_filtered]
        tags_seq = [[self.__get_index(word['tag'], tags2index) for word in sentence] for sentence in
                    sentences_replaced_filtered]
        deps_seq = [[self.__get_index(word['dep'], deps2index) for word in sentence] for sentence in
                    sentences_replaced_filtered]
        positions_seq = [[word['position'] + 1 for word in sentence] for sentence in sentences_replaced_filtered]
        parents_seq = [[word['parent'] + 1 for word in sentence] for sentence in sentences_replaced_filtered]
        subtrees_seq = [[word['subtree_length'] + 1 for word in sentence] for sentence in sentences_replaced_filtered]
        prevs_seq = [[[0, 0, 1]] + [self.__to_one_hot(int(word['retained']), 3) for word in sentence][:-1] for sentence in
                     sentences_replaced_filtered]

        labels = [[self.__to_one_hot(int(word['retained']), 2) for word in sentence] for sentence in
                  sentences_replaced_filtered]

        words_inputs = self.__pad_sequences(words_seq, max_length)
        replaced_words_inputs = self.__pad_sequences(replaced_words_seq, max_length)
        tags_inputs = self.__pad_sequences(tags_seq, max_length)
        deps_inputs = self.__pad_sequences(deps_seq, max_length)
        positions_inputs = self.__pad_sequences(positions_seq, max_length)
        parents_inputs = self.__pad_sequences(parents_seq, max_length)
        subtrees_inputs = self.__pad_sequences(subtrees_seq, max_length)
        prevs_inputs = self.__pad_sequences(prevs_seq, max_length, [0, 0, 0])

        outputs = self.__pad_sequences(labels, max_length, [0, 0])

        np.save('%s/inputs/words_inputs_augmented.npy' % self.data, words_inputs)
        np.save('%s/inputs/replaced_words_inputs_augmented.npy' % self.data, replaced_words_inputs)
        np.save('%s/inputs/tags_inputs_augmented.npy' % self.data, tags_inputs)
        np.save('%s/inputs/deps_inputs_augmented.npy' % self.data, deps_inputs)
        np.save('%s/inputs/positions_inputs_augmented.npy' % self.data, positions_inputs)
        np.save('%s/inputs/parents_inputs_augmented.npy' % self.data, parents_inputs)
        np.save('%s/inputs/subtrees_inputs_augmented.npy' % self.data, subtrees_inputs)
        np.save('%s/inputs/prevs_inputs_augmented.npy' % self.data, prevs_inputs)
        np.save('%s/inputs/outputs_augmented.npy' % self.data, outputs)

    def __load_glove(self, filename, vocab):
        df = pd.read_csv(filename, sep=" ", quoting=3, header=None, index_col=0)
        return {key: val.values for key, val in df.T.items()}

    def __to_glove_matrix(self, glove, vocab, embedding_size=100):
        glove_matrix = [np.zeros(shape=(1, embedding_size), dtype=np.float32)]

        for word in sorted(list(vocab[1:])):
            try:
                vector = glove[word]
                glove_matrix.append(vector.reshape((1, len(vector))))
            except:
                glove_matrix.append(np.random.rand(1, embedding_size))

        return np.concatenate(glove_matrix, axis=0)

    def __save_glove_matrix(self, filename, glove_matrix):
        np.save(filename, glove_matrix)

    def __build_glove_matrix(self):
        words_vocab = load_obj('%s/vocabs/words_vocab_augmented' % self.data)

        print('extracting glove vectors...')
        glove_vectors = self.__load_glove('%s/glove/glove.6B.100d.txt' % self.data, set(words_vocab))
        print('building glove matrix...')
        glove_matrix = self.__to_glove_matrix(glove_vectors, words_vocab)
        print('saving glove matrix...')
        self.__save_glove_matrix('%s/glove/words_glove_matrix_augmented.npy' % self.data, glove_matrix)

        replaced_words_vocab = load_obj('%s/vocabs/replaced_words_vocab_augmented' % self.data)

        print('extracting glove vectors...')
        glove_vectors = self.__load_glove('%s/glove/glove.6B.100d.txt' % self.data, set(replaced_words_vocab))
        print('building glove matrix...')
        glove_matrix = self.__to_glove_matrix(glove_vectors, replaced_words_vocab)
        print('saving glove matrix...')
        self.__save_glove_matrix('%s/embeddings/replaced_words_glove_matrix_augmented.npy' % self.data, glove_matrix)

    def __log(self, current, callback, total_steps):
        print('\r{:.2f}%'.format((current * 100) / total_steps), end='')
        return callback

    def __extract_features_from_word(self, word, label):
        return {
            'position': word.i,
            'text': word.text,
            'lemma': word.lemma_,
            'tag': word.tag_,
            'pos': word.pos_,
            'ent_type': word.ent_type_,
            'ent_iob': word.ent_iob,
            'dep': word.dep_,
            'parent': word.head.i,
            'subtree_length': len(list(word.subtree)),
            'retained': bool(label)
        }

    def __extract_features_from_sentence(self, sentence, labels):
        return [self.__extract_features_from_word(word, label) for word, label in zip(sentence, labels)]

    def __replace_word(self, word):
        word['original'] = word['text']

        if word['ent_type'] != '':
            word['text'] = self.substitute_words[word['ent_type']]
        elif word['lemma'] != '-PRON-':
            word['text'] = word['lemma']

        word['text'] = word['text'].lower()

        return word

    def __replace_words(self, sentence):
        return [self.__replace_word(word) for word in sentence]

    def __get_index(self, word, dict_):
        try:
            return dict_[word]
        except:
            return dict_['<UNK>']

    def __to_one_hot(self, one, length):
        return [1 if i == one else 0 for i in range(length)]

    def __pad_sequences(self, sequences, max_length, pad_token=0):
        return np.array([seq + ([pad_token] * (max_length - len(seq))) for seq in sequences])


class SeparatorTokenizer(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')

        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
