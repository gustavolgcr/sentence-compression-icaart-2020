import tensorflow as tf

from preprocessing import Preprocessor


def main():
    data = "data"
    sentences_file_path = data + "/google/sentence_en.txt"
    compressions_file_path = data + "/google/compression_en.txt"

    preprocessor = Preprocessor(sentences_file_path, compressions_file_path)



if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
