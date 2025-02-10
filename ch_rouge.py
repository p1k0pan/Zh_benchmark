import jieba
from rouge import Rouge 
import numpy as np

def chinese_rouge(references, hypotheses, use_seg=True, rouge_type='rouge-l'):
    """ 中文优化的ROUGE计算 """
    # 预处理函数
    def preprocess(text):
        text = text.replace(" ", "").strip()  # 去空格
        if use_seg:
            return " ".join(jieba.cut(text))  # 分词
        return " ".join(text)  # 按字符分割

    # 初始化ROUGE计算器
    rouge = Rouge(metrics=[rouge_type])
    
    # 预处理所有文本
    # refs = [[preprocess(ref) for ref in ref_group] for ref_group in references]
    refs = [preprocess(ref) for ref in references]
    hyps = [preprocess(hyp) for hyp in hypotheses]

    # 计算分数
    scores = []
    for hyp, ref in zip(hyps, refs):
        group_scores = []
        # for ref in ref_group:
        if len(ref) == 0 or len(hyp) == 0:
            score = {'p': 0.0, 'r': 0.0, 'f': 0.0}
        else:
            try:
                score = rouge.get_scores(hyp, ref)[0][rouge_type]
            except:
                score = {'p': 0.0, 'r': 0.0, 'f': 0.0}
        group_scores.append(score)
        
        # 聚合策略：平均所有参考
        avg_p = np.mean([s['p'] for s in group_scores])
        avg_r = np.mean([s['r'] for s in group_scores])
        avg_f = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        scores.append({'p': avg_p, 'r': avg_r, 'f': avg_f})
    
    return np.mean([s['f'] for s in scores]), scores  # 返回平均F1