import os
import numpy as np
from fairseq.data import Dictionary

import config
from utils import *
from models import EncodeLines, ClassifierModel

vocab = Dictionary()
vocab.add_from_file(os.path.join(config.PATH['PHO_BERT'], "dict.txt"))

classifier = ClassifierModel(config.PATH['PRETRAIN'])
converter = EncodeLines(vocab, classifier.bert_model.bpe)
label_encode = load_pkl(config.PATH['LABEL_ENCODE'])

def predict(file_path, device = 'cpu'):
    '''
        predict label of file
    '''
    with open(file_path, 'r') as f:
        lines = f.read()
        lines_encode = converter.convert([lines])
        idx = classifier.predict(lines_encode)
        return label_encode.inverse_transform([idx])

# subject = predict('./test.txt')
# print("predict !!!!!!!!!!!! ", subject)