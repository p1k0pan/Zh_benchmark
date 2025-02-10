# Zh_benchmark


## 环境
```
pip install rouge jieba bert-score sacrebleu peft==0.12.0 transformers==4.44.2 torch pandas
```

## 代码
代码里自行更改文件地址，注意e1和e2里，json和txt两种文件的有不同的调用方式

e1需要分类器的weights

e2可以安装工具后直接运行

e3需要运行过e1才能运行

## 数据集
toxic_anno_full = 全集毒性文本

toxic_anno_full_rewrite = 全集标准答案

toxic_anno_full = 测试集毒性文本

toxic_anno_full_rewrite = 测试集标准答案