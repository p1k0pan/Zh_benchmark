import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import sys

def ctn_hit(words, target):
    """
    words: pd.Series
    target: iterable
    """
    prefix = "你现在要作为一个毒性文本的分类器，判断以下句子是否适宜展示给用户，判断依据为检测到文本是否含有毒性词，可以忽略整个句子的上下文是否含有攻击性。如果判断适宜展示给用户，请回答是，否则回答否。\n判断的句子如下："
    hit = 0
    ne_hit = 0
    result = []
    assert len(words) == len(target), "Length of words and target not match"
    total = len(target)
    for i in tqdm(range(len(words))):
        flag = False
        cur_words = words[i].split(",")
        ans = target[i]["prompt"].removeprefix(prefix)
        cls = target[i]["cls"]
        for w in cur_words:
            if w in ans:
                flag = True
                result.append({"id": i, "sent":ans, "word": w})
        if flag:
            hit += 1
            ne_hit +=1
        else:
            if cls == "是":
                ne_hit += 1
            result.append({"id": i, "sent":ans, "word": "ne_hit"})
    return total-hit, total-ne_hit, result



if __name__ == "__main__":
    data_file = "/ltstorage/home/2pan/Qwen/data/toxicn/e3_words_test.xlsx"
    data = pd.read_excel(data_file)

    words = data["Words"]
    answers_path = "/ltstorage/home/2pan/Ch_benchmark/results/final/qwen1.5-14b-chat-v2/test_set/prompt3/你是一个改写文本的助手。"
    data_dir = Path(answers_path)
    all_result = []
    # exclude_folder = ["full_zero", "full_few", "llama_few_v2"]
    # exclude_folder = ["test_zero", "test_few", "llama_few_v2"]
    for file in data_dir.rglob("*_e1_cls.json"):  # 递归查找所有的 .txt 文件
        print("process:", file)
        
        data = json.load(open(file, 'r', encoding='utf-8'))
        hit,ne_hit, result = ctn_hit(words, data)
        json.dump(result, open(file.with_name(file.stem+"_e3_words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)

        all_result.append({"model": file.stem,"type":file.parent.name, "hit": hit, "ne_hit": ne_hit, "file": file})
    df = pd.DataFrame(all_result)
    df.to_csv(answers_path+f"/E3_words.csv", index=False, encoding='utf-8-sig' )
