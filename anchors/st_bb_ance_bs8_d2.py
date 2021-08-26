import os
import glob
import time
import argparse
import moxing as mox
import sys

s3_rootdir = "s3://obs-app-2020042019121301221/SEaaKM/m50017495/"
# s3_rootdir = "s3://bucket-852/m50017495/"

mox.file.shift('os', 'mox')
os.makedirs("/cache/anchors")
mox.file.copy_parallel(s3_rootdir + '/code/anchors', '/cache/anchors')

os.system('pip install /cache/anchors/torch-1.8.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install /cache/anchors/torchvision-0.9.0+cu101-cp36-cp36m-linux_x86_64.whl')
os.system('pip install -r /cache/anchors/requirements.txt')
os.system('pip install dgl-cu101')


# s3_model_path = s3_rootdir + "replearn/transformers_models/bert-base-uncased/"
s3_model_ict_path = "s3://obs-app-2020042019121301221/SEaaKM/z00562934/models/ict_onetower_new/"
s3_model_bert_path = s3_rootdir + "replearn/transformers_models/bert-base-uncased/"
s3_model_propw_path = s3_rootdir + "replearn/transformers_models/prop_wiki/"
s3_model_propm_path = s3_rootdir + "replearn/transformers_models/prop_msmarco/"
s3_model_wlp_path = s3_rootdir + "replearn/transformers_models/prop_msmarco/"



s3_train_path = s3_rootdir + "data/anchor_data/finetune/processed_1s/"
s3_dev_path = s3_rootdir + "data/anchor_data/finetune/processed_dev/"

s3_r_train_path = s3_rootdir + "data/anchor_data/finetune/processed_msrerank_1s/"
s3_r_dev_path = s3_rootdir + "data/anchor_data/finetune/processed_msmarco_rerank_dev/"

s3_msmarco_path = s3_rootdir + "data/anchor_data/msmarco/"
s3_output_path = s3_rootdir+ "/output/anchor_output/st_bb_ance_bs8_d2"

s3_dev_trecdl_path = s3_rootdir + "/data/anchor_data/finetune/processed_trecdl_fullrank_eval/"
s3_dev_trecdl_r_path = s3_rootdir + "/data/anchor_data/finetune/processed_trecdl_rerank_eval/"
s3_req_path = s3_rootdir + "/data/requirement/"



bs_predevice_train = 8
bs_predevice_test = 256
epoch = 2
# print("s3_model_path", s3_model_path)
print("s3_train_path", s3_train_path)
print("s3_dev_path", s3_dev_path)
print("s3_msmarco_path", s3_msmarco_path)
print("s3_output_path", s3_output_path)
print("bs_predevice_train", bs_predevice_train)
print("bs_predevice_test", bs_predevice_test)

def extract_data():

	# os.makedirs('/home/work/transformers_models')
	# mox.file.copy_parallel("s3://bucket-852/m50017495/replearn/transformers_models", '/home/work/transformers_models')


	os.makedirs('/cache/mymodel')
	os.makedirs('/cache/mymodel/ict')
	os.makedirs('/cache/mymodel/bert')
	os.makedirs('/cache/mymodel/wlp')
	os.makedirs('/cache/mymodel/propw')
	os.makedirs('/cache/mymodel/propm')
	mox.file.copy_parallel(s3_model_ict_path, '/cache/mymodel/ict')	
	mox.file.copy_parallel(s3_model_bert_path, '/cache/mymodel/bert')	
	mox.file.copy_parallel(s3_model_wlp_path, '/cache/mymodel/wlp')	
	mox.file.copy_parallel(s3_model_propw_path, '/cache/mymodel/propw')	
	mox.file.copy_parallel(s3_model_propm_path, '/cache/mymodel/propm')	

	os.makedirs('/cache/train_data')
	mox.file.copy_parallel(s3_train_path, '/cache/train_data/')

	os.makedirs('/cache/dev_data')
	mox.file.copy_parallel(s3_dev_path, '/cache/dev_data/')

	os.makedirs('/cache/train_r_data')
	mox.file.copy_parallel(s3_r_train_path, '/cache/train_r_data/')

	os.makedirs('/cache/dev_r_data')
	mox.file.copy_parallel(s3_r_dev_path, '/cache/dev_r_data/')

	os.makedirs("/cache/dev_data_trecdl")
	mox.file.copy_parallel(s3_dev_trecdl_path, "/cache/dev_data_trecdl/")

	os.makedirs("/cache/dev_data_trecdl_r")
	mox.file.copy_parallel(s3_dev_trecdl_r_path, "/cache/dev_data_trecdl_r/")
	
	os.makedirs('/cache/msmarco')
	mox.file.copy_parallel(s3_msmarco_path, '/cache/msmarco/')

	os.makedirs('/cache/output')
	os.makedirs('/cache/output/fullrank')
	os.makedirs('/cache/output/rerank')

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

def install_package():
	os.makedirs('/cache/mypackages/')
	mox.file.copy_parallel(s3_req_path, '/cache/mypackages/')	
	os.system("pip install sentencepiece==0.1.90")
	print("begin pytrec")
	os.system("cd /cache/mypackages/pytrec_eval-0.5 && python setup.py install")
	print("pytrec ok")

def main():
	extract_data()
	args = parse_args()

	install_package()

	models = ['bert']
	for m in models:
		
		# print('start training..........')
		print(f"[msmarco fullrank] [{m}] training and evaluating.........")
		os.system(f'cd /cache/anchors/finetune && CUDA_VISIBLE_DEVICES=0,1 python runBert.py \
			--is_training \
			--per_gpu_batch_size {bs_predevice_train} --per_gpu_test_batch_size {bs_predevice_test} --task msmarco \
			--bert_model /cache/mymodel/{m}/ \
			--dataset_script_dir /cache/anchors/data_scripts \
			--dataset_cache_dir /cache/negs_fullrank_msmarco_cache_{m} \
			--log_path /cache/output/fullrank/log_{m}.txt \
			--train_file /cache/train_data \
			--dev_file  /cache/dev_data/all.json \
			--dev_id_file /cache/dev_data/ids.tsv \
			--msmarco_score_file_path /cache/output/fullrank/score_{m}.txt \
			--msmarco_dev_qrel_path /cache/msmarco/msmarco-docdev-qrels.tsv \
			--save_path /cache/output/fullrank/{m}.bin \
			--epochs {epoch} \
			--id {m}_fullrank_msmarco')

		print(f"[trecdl fullrank] [{m}] evaluating.........")
		os.system(f'cd /cache/anchors/finetune && CUDA_VISIBLE_DEVICES=0,1 python runBert.py \
			--per_gpu_batch_size {bs_predevice_train} --per_gpu_test_batch_size {bs_predevice_test} --task trecdl \
			--bert_model /cache/mymodel/{m} \
			--dataset_script_dir /cache/anchors/data_scripts \
			--dataset_cache_dir /cache/negs_fullrank_trecdl_cache_{m} \
			--log_path /cache/output/fullrank/log_{m}_predict.txt \
			--train_file /cache/train_data \
			--dev_file  /cache/dev_data_trecdl/all.json \
			--dev_id_file /cache/dev_data_trecdl/ids.tsv \
			--msmarco_score_file_path /cache/output/fullrank/score_{m}.txt \
			--msmarco_dev_qrel_path /cache/msmarco/trec-2019qrels-docs.txt \
			--save_path /cache/output/fullrank/{m}.bin \
			--epochs {epoch} \
			--id {m}_fullrank_trecdl')
    
		mox.file.copy_parallel('/cache/output', s3_output_path)
		print("write success")




if __name__ == '__main__':
	main()
