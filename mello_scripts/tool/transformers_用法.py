from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("ywan/unite-up", num_labels=1)

# 模型保存到指定路径并加载
model = AutoModelForSequenceClassification.from_pretrained("ywan/unite-up", num_labels=1, cache_dir='./working_dir')
model = AutoModelForSequenceClassification.from_pretrained('./working_dir', num_labels=1)

device = torch.device(^^^^)
src = tokenizer([list of sents], xxxxx)

# 输入只有一个句子的情况，batch=1

## src = tokenizer(sent,padding,truncation,max_length)
## src:dict:{
##   "input_ids":list of float
##   "attention_mask":list of float  
## }

## 铭铭代码的写法
## src = tokenizer(sent,padding,truncation,max_length, return_tensors="pt")
## src:dict:{
##   "input_ids":tensor:tensor.float64
##   "attention_mask":tensor 
## }

## src["input_ids"]
## src["attention_mask"]

model.to(device)
output = model(src["input_ids"].to(device), attention_mask=src["attention_mask"].to(device))

# 输入只有一个句子的情况，batch=batch_size
src = tokenizer([list of sent],padding,truncation,max_length)
## len([list of sent])=batch_size
output = model(src["input_ids"].to(device), attention_mask=src["attention_mask"].to(device))

# 输入有一对句子的情况, batch=1
src = tokenizer(sent_1,sent_2,padding,truncation,max_length)
output = model(src["input_ids"].to(device), attention_mask=src["attention_mask"].to(device))

# 输入有一对句子的情况, batch=batch_size
src = tokenizer([list_1 of sent],[list_2 of sent],padding,truncation,max_length)
## [list_1 of sent],[list_2 of sent]的长度必须是一样的

#####

# 输入有src，ref，hyp三个句子的情况，其中：
# src = '你好！'
# ref = 'Hello!'
# hyp = 'Hi!'

## tokenizer.cls_token + tokenizer.sep_token / tokenizer.bos_token + tokenizer.eos_token（有的模型用cls+sep，有的用bos+eos，要视具体情况决定）
## 看法：在terminal里尝试输出tokenizer.cls_token
input_in_str = tokenizer.cls_token + src + tokenizer.sep_token + ref + tokenizer.sep_token + hyp + tokenizer.sep_token
input_in_tensor = tokenizer(input_in_str, padding="max_length", max_length=300, truncation=True, add_special_tokens=False, return_tensors="pt")

output = model(src["input_ids"].to(device), attention_mask=src["attention_mask"].to(device))

# 导出logits和loss
logits = output.logits
loss = output.loss

XXXXXXXXXXX