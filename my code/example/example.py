import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras.utils import plot_model   # 模型可視化


def load_data():
    train = _parse_data(open(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\data\train_data.data','rb'))
    test = _parse_data(open(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\data\test_data.data', 'rb'))
    # print(test)
    word_counts = Counter(row[0].lower() for sample in train for row in sample)

    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]

    chunk_tags = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]

    # save initial config data
    with open(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\config.pkl', 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    test = _process_data(test, vocab, chunk_tags)
    # print(train)
    # print('\n')
    # print(test)
    # print('\n')
    # print(chunk_tags)
    # print('\n')
    return train, test, (vocab, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . 
    #  so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    # set to <unk> (index 1) if not in vocab
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=100):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

EMBED_DIM = 200
BiRNN_UNITS = 200


def create_model(train=True):
    if train:
        (train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = load_data()
        # print(len(train_x))
        # print(len(train_y))
        # print(vocab, chunk_tags)
    else:
        with open(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\config.pkl', 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)

    model = Sequential()
    print(len(vocab))
    # model.add(Embedding(len(vocab), EMBED_DIM,mask_zero = False ))  # Random embedding
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero = False))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (vocab, chunk_tags)


# EPOCHS = 10

# model, (train_x, train_y), (test_x, test_y) = create_model()
# # train model
# print(train_x)
# print(len(train_x))
# model.fit(train_x, train_y,batch_size=16,epochs=EPOCHS, validation_data=[test_x, test_y])

# model.save(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\crf.h5')

model, (vocab, chunk_tags) = create_model(train=False)
predict_text = '中华人民共和国国务院总理周恩来在外交部长陈毅的陪同下，连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚'
str, length = process_data(predict_text, vocab)
print(str)
model.load_weights(r'C:\Users\admin\Desktop\自然語言處理\zh-NER-keras-master\crf.h5')
raw = model.predict(str)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

per, loc, org = '', '', ''

for s, t in zip(predict_text, result_tags):
    if t in ('B-PER', 'I-PER'):
        per += ' ' + s if (t == 'B-PER') else s
    if t in ('B-ORG', 'I-ORG'):
        org += ' ' + s if (t == 'B-ORG') else s
    if t in ('B-LOC', 'I-LOC'):
        loc += ' ' + s if (t == 'B-LOC') else s

print(['person:' + per, 'location:' + loc, 'organzation:' + org])