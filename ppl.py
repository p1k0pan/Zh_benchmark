import torch
from tqdm import tqdm
def token_loss(lm_logits, labels):
    bsz = lm_logits.size(0)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(bsz, -1)
    return loss

def ppl_eval(texts, labels, model, tokenizer):
    for text in tqdm(texts, position=0, leave=True):

        inputs = tokenizer(text, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        labels = inputs['input_ids'].clone()
        # ignore pad token
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            logits = model(**inputs).logits

        loss = token_loss(logits, labels)
        ppls = (loss.sum(dim=-1) / (loss != 0).sum(dim=-1)).exp().cpu().tolist()
    return ppls


def calculate_chinese_ppl(texts, labels, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载中文专用模型
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # model.eval()
    
    ppls = []
    for i, text in enumerate(tqdm(texts, desc="Calculating PPL")):
        inputs = tokenizer(labels[i], return_tensors="pt").to(device)
        # label_inputs = tokenizer(labels[i], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids.clone())
        ppl = torch.exp(outputs.loss).item()
        if ppl < 1e4:  # for sanity
            ppls.append(ppl)
    return ppls

