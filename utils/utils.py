from pickle import dump, load


def save_obj(obj, filepath):
    dump(obj, open(filepath, 'wb'))


def load_obj(filepath):
    return load(open(filepath, 'rb'))


def split_train_dev_test(data, train, dev):
    train_split = train
    dev_split = train - dev

    if len(data.shape) == 2:
        return data[:dev_split, :], data[dev_split:train_split, :], data[train_split:, :]

    if len(data.shape) == 3:
        return data[:dev_split, :, :], data[dev_split:train_split, :, :], data[train_split:, :, :]
