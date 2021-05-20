import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from analyze_dataset import AnalyzeDataset
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test_file_path", default="", type=str)
parser.add_argument("--bert_model_path", default="", type=str)
parser.add_argument("--cl_model_path", default="", type=str)

args = parser.parse_args()
test_batch_size = 128 * torch.cuda.device_count()
device = torch.device("cuda:0")

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

set_seed()
tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
additional_tokens = 3
tokenizer.add_tokens("[eos]")
tokenizer.add_tokens("[term_del]")
tokenizer.add_tokens("[sent_del]")

bert_model = BertModel.from_pretrained(args.bert_model_path)
bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)

model_state_dict = torch.load(args.cl_model_path)
bert_model.load_state_dict({k.replace('bert_model.', ''):v for k, v in model_state_dict.items()}, strict=False)

bert_model = bert_model.to(device)
bert_model = torch.nn.DataParallel(bert_model)

bert_model.eval()

test_dataset = AnalyzeDataset(args.test_file_path, 128, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

# batch = test_dataset.__getitem__(19)
# print(batch['input_ids'])
# print(batch['attention_mask'])
# print(batch['token_type_ids'])
# print(batch['labels'])
# assert False

result = {}
all_count = 0

with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader)
    for i, test_data in enumerate(epoch_iterator):
        with torch.no_grad():
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
        input_ids = test_data["input_ids"]
        attention_mask = test_data["attention_mask"]
        token_type_ids = test_data["token_type_ids"]
        input_ids2 = test_data["input_ids2"]
        attention_mask2 = test_data["attention_mask2"]
        token_type_ids2 = test_data["token_type_ids2"]

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        bert_rep =  bert_model(**bert_inputs)[1]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        bert_rep2 =  bert_model(**bert_inputs2)[1]
        bert_norm1 = bert_rep.norm(dim=-1, keepdim=True)  # [batch]
        bert_norm2 = bert_rep2.norm(dim=-1, keepdim=True)  # [batch]
        cossim = torch.einsum("bd,bd->b", bert_rep, bert_rep2) / ((bert_norm1 * bert_norm2) + 1e-6)
        cossim = cossim.data.cpu().numpy().reshape(-1)
        for sim in cossim:
            if sim == 1.0:
                bucket_idx = 40
            else:
                bucket_idx = int((sim + 1) / 0.05) + 1
            result[bucket_idx] = result.get(bucket_idx, 0) + 1
            all_count += 1

print(result)