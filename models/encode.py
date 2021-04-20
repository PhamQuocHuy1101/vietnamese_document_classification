import config
import numpy as np

class EncodeLines():
    def __init__(self, vocab, bpe, max_sequence_length = config.MODEL['SEQUENCE_LENGTH']):
        self.vocab = vocab
        self.bpe = bpe
        self.max_sequence_length = max_sequence_length
    def convert(self, lines):
        outputs = np.zeros((len(lines), self.max_sequence_length), dtype=np.int32)
        eos_id = 2
        pad_id = 1

        for idx, row in enumerate(lines):

            subwords = self.bpe.encode('<s> '+ row +' </s>')
            input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

            if len(input_ids) > self.max_sequence_length: 
                input_ids = input_ids[:self.max_sequence_length] 
                input_ids[-1] = eos_id
            else:
                input_ids = input_ids + [pad_id, ]*(self.max_sequence_length - len(input_ids))

            outputs[idx,:] = np.array(input_ids)
        return outputs
