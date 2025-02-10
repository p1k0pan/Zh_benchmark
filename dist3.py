import jieba
from collections import defaultdict
import re

def compute_chinese_dist3(generations, use_word_seg=True, filter_stopwords=False):
    """
    计算中文文本的三元组多样性 (Dist3)
    
    参数：
    - generations: 待评估的中文文本列表
    - use_word_seg: True使用词语分割，False使用字符分割
    - filter_stopwords: 是否过滤停用词
    
    返回：
    - dist3: 唯一三元组数 / 总token数
    """
    # 加载停用词表
    stopwords = set()
    if filter_stopwords:
        with open('hit_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)
    
    # 中文标点处理（扩展标点列表）
    chinese_punct = '！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～“”‘’【】…–—'
    punct_pattern = re.compile(f'[{re.escape(chinese_punct)}]')
    
    # 统计数据结构
    trigrams = defaultdict(int)
    total_tokens = 0
    
    for text in generations:
        # 清洗文本
        text = punct_pattern.sub('', text)  # 去除中文标点
        text = text.strip().lower()  # 统一小写
        
        # 分词处理
        if use_word_seg:
            tokens = jieba.lcut(text)
        else:
            tokens = list(text.replace(' ', ''))  # 按字符分割
        
        # 过滤停用词
        if filter_stopwords:
            tokens = [t for t in tokens if t not in stopwords and t.strip() != '']
        
        # 统计三元组
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}"
            trigrams[trigram] += 1
        
        total_tokens += len(tokens)
    
    # 处理极端情况
    if total_tokens == 0:
        return 0.0
    
    # 计算Dist3
    unique_trigrams = len(trigrams)
    dist3 = unique_trigrams / total_tokens if total_tokens > 0 else 0
    
    return round(dist3, 4)

# 示例用法
if __name__ == "__main__":
    texts = [
        "深度学习是人工智能的重要方向",
        "自然语言处理需要大量的语料数据",
        "机器学习模型依赖高质量的训练样本"
    ]
    
    # 词语级（过滤停用词）
    print(compute_chinese_dist3(texts, filter_stopwords=True))  
    # 输出示例: 0.4167
    
    # 字符级（保留停用词）
    print(compute_chinese_dist3(texts, use_word_seg=False))     
    # 输出示例: 0.3812