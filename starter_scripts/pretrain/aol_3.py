import os
import glob
import time
import argparse
import moxing as mox
import sys

s3_rootdir = "s3://obs-app-2020042019121301221/SEaaKM/m50017495/"
# s3_rootdir = "s3://bucket-852/m50017495/"

mox.file.shift('os', 'mox')
os.makedirs("/cache/ss")
mox.file.copy_parallel(s3_rootdir + '/code/session_search', '/cache/ss')

os.system('pip install /cache/ss/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/ss/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')

os.system('pip install -r /cache/ss/requirements.txt')
os.system('pip install dgl-cu101')


s3_model_path = s3_rootdir + "/data/yutao/"
s3_req_path = s3_rootdir + "/data/requirement/"
s3_output_path = s3_rootdir + "/output/yutao/aol_3/"


def install_package():
    os.makedirs('/cache/mypackages/')
    mox.file.copy_parallel(s3_req_path, '/cache/mypackages/')
    os.system("pip install sentencepiece==0.1.90")
    print("begin pytrec")
    os.system("cd /cache/mypackages/pytrec_eval-0.5 && python setup.py install")
    print("pytrec ok")


def extract_data():
    os.makedirs('/cache/data')
    mox.file.copy_parallel(s3_model_path, '/cache/data')

    os.makedirs('/cache/output')
    os.makedirs('/cache/output/pretraining')
    os.makedirs('/cache/output/pointwise')
    os.makedirs('/cache/output/pretraining/logs')
    os.makedirs("/cache/output/pretraining/models")
    os.makedirs('/cache/output/pointwise/logs')
    os.makedirs("/cache/output/pointwise/models")
    

def parse_args():
    parser = argparse.ArgumentParser(description='Process Reader Data')
    # to ignore
    parser.add_argument('--data_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='data_url for yundao')
    parser.add_argument('--init_method', default='',
                        help='init_method for yundao')
    parser.add_argument('--train_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='train_url for yundao')
    parser.add_argument("--s3_path_dir", type=str,
                        default='s3://bucket-852/f00574594/data/HGN_data/train_data_with_tfidf30_bert_large_aug/path_data/',
                        help='define path directory')
    parser.add_argument("--s3_HGN_data_dir", type=str,
                        default='s3://bucket-852/f00574594/data/KFB_data/reader_data_no_sep/',
                        help='define output directory')
    parser.add_argument("--my_output_dir", type=str,
                        default='s3://bucket-852/m50017495/replearn/output_train/',
                        help='define output directory')
    return parser.parse_args()


def main():
    extract_data()
    args = parse_args()

    install_package()
    
    os.system(f"cd /cache/ss/Pretraining && CUDA_VISIBLE_DEVICES=0,1,2,3 python runBertContras.py --bert_model_path /cache/data/BertModel/ --per_gpu_batch_size 128 --log_path /cache/output/pretraining/logs/ --save_path /cache/output/pretraining/models/ --epochs 3 --temperature 0.1 --aug_strategy sent_deletion,term_deletion,qd_reorder --ratio 0.6")
    os.system(f"cd /cache/ss/Pointwise && python runBert.py --score_file_path /cache/output/BertContrastive.aol.3.10.128.sent_deletion.term_deletion.qd_reorder.score.txt --bert_model_path /cache/data/BertModel/ --per_gpu_batch_size 64 --log_path /cache/output/pointwise/logs/ --save_path /cache/output/pointwise/models/bert_sessionsearch.aol.3.10.128.sent_deletion.term_deletion.qd_reorder --epochs 3 --pretrain_model_path /cache/output/pretraining/models/BertContrastive.aol.3.10.128.sent_deletion.term_deletion.qd_reorder --learning_rate 5e-5")

    mox.file.copy_parallel('/cache/output', s3_output_path)
    print("write success")




if __name__ == '__main__':
    main()
