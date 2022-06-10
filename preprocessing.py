from nltk.tokenize import word_tokenize


class Preprocessor:
    def __init__(self, sentences_file_path, compressions_file_path):
        self.sentences_file_path = sentences_file_path
        self.compressions_file_path = compressions_file_path
        self.sentences = []
        self.compressions = []
        self.sentences_tokenized = []
        self.compressions_tokenized = []

    def preprocess_data(self):
        self.__data_import()
        self.__tokenizer()

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
