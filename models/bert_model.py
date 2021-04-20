import torch
import os
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

import config

class BPE():
    bpe_codes = os.path.join(config.PATH['PHO_BERT'], 'bpe.codes')

# Wrapper class
class ClassifierModel:
    def __init__(self, pretrain_path = None, n_class = config.MODEL['N_CLASS'], device = 'cpu'):
        super(ClassifierModel, self).__init__()
        self.device = device

        from fairseq.models.roberta import RobertaModel
        self.bert_model = RobertaModel.from_pretrained(config.PATH['PHO_BERT'], checkpoint_file='model.pt')
        self.bert_model.bpe = fastBPE(BPE())
        self.bert_model.register_classification_head('new_task', num_classes = n_class)
        self.bert_model.to(device = device)

        if pretrain_path != None:
            self.load_model(pretrain_path)
        self.bert_model.eval()

    def load_model(self, pretrain_path):
        checkpoint = torch.load(pretrain_path, map_location = self.device)
        self.bert_model.load_state_dict(checkpoint['state_dict'])
        print("Classifier model load pretrain success!")
    
    def predict(self, lines_encode):
        self.bert_model.eval()
        with torch.no_grad():
            logits = self.bert_model.predict('new_task', torch.tensor(lines_encode, dtype = torch.long, device = self.device))
            return torch.argmax(logits.squeeze())
