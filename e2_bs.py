
import pandas as pd
from tqdm.contrib import tzip
from pathlib import Path
import os
from bert_score import score
import torch
import sacrebleu
from ch_rouge import chinese_rouge
from dist3 import compute_chinese_dist3
from ppl import calculate_chinese_ppl, ppl_eval
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json

def bleu_score(predict, answer, chinese=True):
    """
    refs = [ 
             ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'], 
           ]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    """
    if chinese:
        # predict2 = [" ".join(jieba.cut(p, cut_all=False)) for p in predict]
        # answer2 = [[" ".join(jieba.cut(a, cut_all=False)) for a in answer[0]]]
        # bleu = sacrebleu.corpus_bleu(predict2, answer2, lowercase=True, tokenize="none")
        bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="zh")
    else:
        bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="13a")
    return bleu.score

def bertscore(predict, answer, chinese=False):
    P, R, F1 = score(predict, answer, lang = "zh", device="cuda")
    return torch.mean(P).item(), torch.mean(R).item(), torch.mean(F1).item()

def cal_total(predict, answer):
    print("calculating BERTScore")
    p, r, f1 = bertscore(predict, answer)
    print("calculating BLEU")
    bs = bleu_score(predict, [answer])
    print("calculating ROUGE")
    rg,_ = chinese_rouge(answer, predict)
    print("calculating Dist3")
    d3 = compute_chinese_dist3(predict)
    print("calculating PPL")
    pl_score = calculate_chinese_ppl(predict, answer, model, tokenizer)
    # pl_score = ppl_eval(predict, answer, model, tokenizer)
   
    return p, r, f1, bs, rg, d3, np.mean(pl_score)

def eval_csv():
    rewrite_file = "qwen_sft/SFT_outcome.xlsx"
    sheets = ['Qwen1.5_7b_Chat', 'Qwen1.5_14b_Chat', "Baichuan2_7b_Chat", "Baichuan2_13b_Chat", 
              "Llama2_7b_Chat", "Llama2_13b_Chat", "ChatGLM3_6b"]
    for sheet_name in sheets:
        try:
            print(sheet_name)
            data = pd.read_excel(rewrite_file, sheet_name=sheet_name)
        except:
            continue
        # eval_tox_df(data, path+sheet_name)
        with open(standard_file_path, 'r', encoding='utf-8') as f:
            standard_texts = f.readlines()
        
        answer=[]
        predict=[]
        for i in range(len(standard_texts)):
            answer.append(standard_texts[i].strip())
            predict.append(data["SFT"][i].strip())
        try:
            p, r, f1, bs, rg, d3, pl_score = cal_total(predict, answer)
        except:
            print("error in:", sheet_name)
            continue
        all_result.append({"model": sheet_name, "type":"SFT", "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "BLEU": bs, "ROUGE": rg, "Dist3": d3, "PPL": pl_score,  "file": file})


def eval_txt():
    exclude_folder = ["full_few", "full_zero"]  
    for file in data_dir.rglob("*.txt"):  # 递归查找所有的 .txt 文件
        if file.parts[-2] in exclude_folder:
            continue  # 跳过包含排除文件夹的文件
        print("process:", file)

        # similarities, gt50 = e2_cal(standard_file_path, file)
        with open(standard_file_path, 'r', encoding='utf-8') as f:
            standard_texts = f.readlines()
        with open(file,'r', encoding='utf-8') as f:  # 使用 Path 对象的 open 方法
            pred_texts = f.readlines()
        
        predict=[]
        answer=[]
        for i in range(len(standard_texts)):
            answer.append(standard_texts[i].strip())
            predict.append(pred_texts[i].strip())
        try:
            p, r, f1, bs, rg, d3, pl_score = cal_total(predict, answer)
        except Exception as e:
            print("error in:", file, e)
            continue

        # df = pd.DataFrame(similarities)
        # df.to_csv(file.with_name(file.stem+"_E2_bs.csv"), index=False, encoding='utf-8-sig')
        # all_result.append({"model": file.stem, "type":file.parts[-2], "sim>0.5": gt50}) 
        all_result.append({"model": file.stem, "type":file.parts[-2], "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "BLEU": bs, "ROUGE": rg, "Dist3": d3, "PPL": pl_score,  "file": file}) 
    return all_result

def eval_json():
    
    for file in data_dir.rglob("*b.json"):  # 递归查找所有的 .txt 文件
        print("process:", file)

        # similarities, gt50 = e2_cal(standard_file_path, file)
        with open(standard_file_path, 'r', encoding='utf-8') as f:
            standard_texts = f.readlines()
        pred_texts = json.load(open(file, 'r', encoding='utf-8'))
        
        predict=[]
        answer=[]
        for i in range(len(standard_texts)):
            answer.append(standard_texts[i].strip())
            predict.append(pred_texts[i]["pred"].strip())
        p, r, f1, bs, rg, d3, pl_score = cal_total(predict, answer)

        # df = pd.DataFrame(similarities)
        # df.to_csv(file.with_name(file.stem+"_E2_bs.csv"), index=False, encoding='utf-8-sig')
        # all_result.append({"model": file.stem, "type":file.parts[-2], "sim>0.5": gt50}) 
        all_result.append({"model": file.stem, "type":file.parts[-2], "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "BLEU": bs, "ROUGE": rg, "Dist3": d3, "PPL": pl_score, "file": file}) 

if __name__ == "__main__":
    all_dict = {
                "/mnt/data/users/liamding/data/Qwen/data/annotations/toxic_anno_test_rewrite.txt": 
                "tox_eval_co/",
                # "/ltstorage/home/2pan/Qwen/data/toxicn/annotations/toxic_anno_full_rewrite.txt": 
                # "final/rewritten_results",
                }
    model_dir = "/mnt/data/users/liamding/data/models/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    # all_result = eval_txt()
    for standard_file_path, data_dir in all_dict.items():
        data_dir = Path(data_dir)
        all_result = []
        eval_json()
        # eval_txt()
        # eval_sft()

    
    res_df = pd.DataFrame(all_result)
    res_df.to_csv(os.path.join(data_dir, "metrics_e2.csv"), index=False, encoding='utf-8-sig')
