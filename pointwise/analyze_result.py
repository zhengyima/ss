import numpy as np
import pytrec_eval
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--score_file_path", default="", type=str)
parser.add_argument("--test_file_path", default="", type=str)
args = parser.parse_args()

def __read_socre_file(score_file_path):
    sessions = []
    one_sess = []
    with open(score_file_path, 'r') as infile:
        i = 0
        for line in infile.readlines():
            i += 1
            tokens = line.strip().split('\t')
            one_sess.append((float(tokens[0]), int(float(tokens[1]))))
            if i % 50 == 0:
                one_sess_tmp = np.array(one_sess)
                if one_sess_tmp[:, 1].sum() > 0:
                    sessions.append(one_sess)
                one_sess = []
    return sessions

def __read_test_file(test_file_path):
    all_len = []
    with open(test_file_path, "r") as fr:
        for idx, line in enumerate(fr):
            line = line.strip().split("\t")[1:]
            if idx % 50 == 0:
                assert len(line) % 2 == 0
                all_len.append(int(len(line) // 2))
    return all_len

def evaluate_all_metrics():
    sessions = __read_socre_file(args.score_file_path)
    all_lens = __read_test_file(args.test_file_path)
    assert len(all_lens) == len(sessions)
    # [(score1, label1), (score2, label2), (score3, label3), ...]
    count_result = {"short": [], "medium": [], "long": []}
    count_result_2 = {"short": [], "medium": [], "long": []}
    for idx, sess in enumerate(sessions):
        qrels = {}
        run = {}
        query_id = str(idx)
        if query_id not in qrels:
            qrels[query_id] = {}
        if query_id not in run:
            run[query_id] = {}
        for jdx, r in enumerate(sess):
            doc_id = str(jdx)
            qrels[query_id][doc_id] = int(r[1])
            run[query_id][doc_id] = float(r[0])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'recip_rank', 'ndcg_cut.1,3,5,10'})
        res = evaluator.evaluate(run)
        map = [v['map'] for v in res.values()]
        assert len(map) == 1
        mrr = [v['recip_rank'] for v in res.values()]
        ndcg_1 = [v['ndcg_cut_1'] for v in res.values()]
        ndcg_3 = [v['ndcg_cut_3'] for v in res.values()]
        ndcg_5 = [v['ndcg_cut_5'] for v in res.values()]
        ndcg_10 = [v['ndcg_cut_10'] for v in res.values()]
        lens = all_lens[idx]
        if lens <= 2:
            count_result["short"].append(map[0])
            count_result_2["short"].append(ndcg_1[0])
        elif lens < 5:
            count_result["medium"].append(map[0])
            count_result_2["medium"].append(ndcg_1[0])
        else:
            count_result["long"].append(map[0])
            count_result_2["long"].append(ndcg_1[0])
    print(np.average(count_result["short"]), np.average(count_result_2["short"]), len(count_result["short"]))
    print(np.average(count_result["medium"]), np.average(count_result_2["medium"]), len(count_result["medium"]))
    print(np.average(count_result["long"]), np.average(count_result_2["long"]), len(count_result["long"]))

evaluate_all_metrics()