import os
import numpy as np
from fairseq.data import Dictionary

import config
from utils import *
from models import EncodeLines, ClassifierModel

vocab = Dictionary()
vocab.add_from_file(os.path.join(config.PATH['PHO_BERT'], "dict.txt"))

print("Loading model ...")
classifier = ClassifierModel(config.PATH['PRETRAIN'])
converter = EncodeLines(vocab, classifier.bert_model.bpe)
label_encode = load_pkl(config.PATH['LABEL_ENCODE'])
print("### ")
def predict(text, device = 'cpu'):
    lines_encode = converter.convert([text])
    idx = classifier.predict(lines_encode)
    return label_encode.inverse_transform([idx])

def predict_from_file(file_path, device = 'cpu'):
    '''
        predict label of file
    '''
    with open(file_path, 'r') as f:
        lines = f.read()
        return predict(lines)
    return None

# subject = predict('./test.txt')
# print("predict !!!!!!!!!!!! ", subject)