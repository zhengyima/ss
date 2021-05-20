import linecache
from torch.utils.data import Dataset
import numpy as np

class AnalyzeDataset(Dataset):
    def __init__(self, filename, max_seq_length, tokenizer):
        super(AnalyzeDataset, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
    
    def check_length(self, pairlist):
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 1:
            pairlist.pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist
    
    def anno_main(self, qd_pairs):
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history_toks = ["[CLS]"]
        for iidx, sent in enumerate(all_qd):
            history_toks.extend(sent + ["[eos]"])
        history_toks += ["[SEP]"]
        all_qd_toks = history_toks
        segment_ids = [0] * len(history_toks)
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        label = int(line[0])
        qd_pairs = line[1:]
        history = qd_pairs[:-1]
        doc = [qd_pairs[-1]]
        input_ids, attention_mask, segment_ids = self.anno_main(history)
        input_ids2, attention_mask2, segment_ids2 = self.anno_main(doc)
        batch = {
            'input_ids': input_ids, 
            'token_type_ids': segment_ids, 
            'attention_mask': attention_mask, 
            'input_ids2': input_ids2, 
            'token_type_ids2': segment_ids2, 
            'attention_mask2': attention_mask2, 
            'labels': float(label)
        }
        return batch
    
    def __len__(self):
        return self._total_data

