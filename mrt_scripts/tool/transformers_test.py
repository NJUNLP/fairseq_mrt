from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer
import torch

# 初次下载，保存到cache_dir
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
# model = BertModel.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(save_directory = 'transformers/xlm-roberta-large')

# 后续直接从路径读取ckpt


# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

# python3 mrt_scripts/tool/transformers_test.py