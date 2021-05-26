import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from file_preprocess_dataset import ContrasDataset
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

test_dataset = ContrasDataset(args.test_file_path, 128, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)

def lalign(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha)

def lunif(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

result = {}
all_count = 0

with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader)
    all_results1 = []
    all_results2 = []
    for i, test_data in enumerate(epoch_iterator):
        with torch.no_grad():
            for key in test_data.keys():
                test_data[key] = test_data[key].to(device)
        input_ids = test_data["input_ids1"]
        attention_mask = test_data["attention_mask1"]
        token_type_ids = test_data["token_type_ids1"]
        input_ids2 = test_data["input_ids2"]
        attention_mask2 = test_data["attention_mask2"]
        token_type_ids2 = test_data["token_type_ids2"]

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        bert_rep =  bert_model(**bert_inputs)[1].data.cpu().numpy()  # [bsz, hidden] 
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        bert_rep2 =  bert_model(**bert_inputs2)[1].data.cpu().numpy()  # [bsz, hidden]

        all_results1.append(bert_rep)
        all_results2.append(bert_rep2)

    all_results1 = np.concatenate(all_results1, axis=0)  # [num, hidden]
    all_results2 = np.concatenate(all_results2, axis=0)  # [num, hidden]

    # all_results = np.concatenate([all_results1, all_results2], axis=0)  # [num * 2, hidden]
    # bert_sne = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(all_results)
    # bert_sne = PCA(n_components=2, random_state=0).fit_transform(all_results)

    # bert_sne1 = bert_sne[:len(bert_sne) // 2]
    # bert_sne2 = bert_sne[len(bert_sne) // 2:]

    # bert_sne1 = torch.tensor(bert_sne1).cuda()
    # bert_sne2 = torch.tensor(bert_sne2).cuda()

    all_results1 = torch.tensor(all_results1).cuda()
    all_results2 = torch.tensor(all_results2).cuda()

    alignment = lalign(all_results1, all_results2).data.cpu().numpy().reshape(-1)
    uniform = 0.5 * lunif(all_results1).data.cpu().numpy().reshape(-1) + 0.5 * lunif(all_results2).data.cpu().numpy().reshape(-1)

    print(alignment.mean())
    print(uniform.mean())

    # for ali in alignment:
    #     bucket_idx = int(ali / 0.25)
    #     result[bucket_idx] = result.get(bucket_idx, 0) + 1
    #     all_count += 1

    # result = sorted(result.items(), key=lambda x: x[0])
    # print(result)