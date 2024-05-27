#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import re
import math
import random
from typing import List
import pandas as pd
import numpy as np
import argparse
import torch
import datasets
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.generation import GenerationConfig
from datasets import load_dataset, load_from_disk


class args:
    checkpoint_path = '/gemini/code/lamma3_eval/lamma3_model/8B_instruct'
    eval_data_path = '/gemini/code/lamma3_eval/eval_data/gsm8k'
    save_result_dir = "/gemini/code/lamma3_eval/eval_result/gsm8k_chat"
    # choices = ["A", "B", "C", "D"]
    debug = False
    overwrite = False
    batch_size = 6


# In[2]:


# dataset = datasets.load_dataset("gsm8k",'main')
# dataset.save_to_disk(args.eval_data_path)

dataset = load_from_disk(args.eval_data_path)


# In[3]:


dataset['test'][0]


# In[4]:


def load_models_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side='left'
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map="auto",
        # quantization_config=quantization_config
        # torch_dtype=torch.bfloat16
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


# In[5]:


model, tokenizer = load_models_tokenizer()
tokenizer.pad_token_id = tokenizer.eos_token_id


# In[6]:


fewshot_prompt = open("gsm8k_prompt.txt").read()
prompt = "Here is 8 examples, please answer the last question in this format.\n\n"
start_prompt = "\nPay attention! You need give your answer at the end in the form of 'The answer is '\nHere is my question:\n"


def batch_process(func, *args):
    '''
    args 负责接受 一个或多个 batch
    '''
    # print(f'args len: {len(args)}')
    
    text_ls = []
    for sample in zip(*args):
        text_ls.append(func(*sample))
    return text_ls


def doc_to_text_item(doc):
    return (
        prompt
        + fewshot_prompt
        + start_prompt
        + "\nQuestion: "
        + doc
        + "\nLet's think step by step\n"
    )

def doc_to_text(doc):
    return batch_process(doc_to_text_item, doc)


# In[8]:


def clear_output_item(text):  # 可能需要改进，可以把raw_text长度传进来，直接截断，然后提取第一个回答。这里相当 取了最后一个回答
    left_extract = "<|start_header_id|>assistant<|end_header_id|>"
    st = text.rfind(left_extract) + len(left_extract)
    output = text[st:]

    stop_words = ["<|end_of_text|>", "<|eot_id|>"]
    for sw in stop_words:
        output = output.replace(sw, "").strip()
    
    if output == "": print(text)
    return output

def clear_output(text):
    return batch_process(clear_output_item, text)


def generate_sample(model, tokenizer, input_txt):
    chat_template = [[{'content': t, 'role': 'user'}] for t in input_txt]
    input_txt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_txt, padding=True, return_tensors="pt").to(model.device)

    # print(input_ids['input_ids'])
    # print(input_ids['attention_mask'])

    outputs_id = model.generate(**input_ids, max_new_tokens = 512, eos_token_id = 128009, pad_token_id = tokenizer.pad_token_id,
                                repetition_penalty = 1.2, do_sample = False, temperature = 1.0, top_p = 1.0)
    # print(outputs)
    outputs = tokenizer.batch_decode(outputs_id, skip_special_tokens=False)
    answer = clear_output(outputs)

    return answer


# In[9]:


def extract_answer_item(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        # print(f"The last digit in {s} is {last_digit}")
    else:
        last_digit = None
        # print(f"No digits found in {s!r}", flush=True)
    return last_digit

def extract_answer(s):
    return batch_process(extract_answer_item, s)


def is_correct_item(completion, answer):
    predict = extract_answer_item(completion)
    gold = extract_answer_item(answer)

    assert gold is not None, "No ground truth answer found in the document."

    if predict is None:
        return False
    try:
        return math.isclose(eval(gold), eval(predict), rel_tol=0, abs_tol=1e-4)
    except:
        print(
            f"cannot compare two numbers: answer={gold}, pred={predict}", flush=True
        )
        return False
    

def is_correct(completion, answer):
    return batch_process(is_correct_item, completion, answer)


# In[10]:


@torch.no_grad()
def main():
    all_sz = len(dataset['test'])
    # test = dataset['test'].select(random.sample(range(all_sz),150))
    test = dataset['test']
    acc_ls = []

    # 看有无文件，有的话就不重复做了
    result_path = os.path.join(args.save_result_dir, f"result.csv")
    if not args.overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        for (_, resultrow) in pd.read_csv(result_path).iterrows():
            # pred = extract_answer(resultrow['model_response'], datarow)
            acc = resultrow["ACC"]
            acc_ls.append(acc)

    else:
        question_ls = []
        answer_ls = []
        response_ls = []

        for i in tqdm(range(0, len(test), args.batch_size)):
            batch = test.select(range(i, min(i+args.batch_size, len(test))))

            context = doc_to_text(batch["question"])
            question_ls.extend(batch["question"])
            # print(context)
            completion = generate_sample(model, tokenizer, context)
            response_ls.extend(completion)
            
            acc = is_correct(completion, batch["answer"])
            answer_ls.extend(batch["answer"])
            acc_ls.extend(acc)

        # 存入文件
        output_df = pd.DataFrame(
            {"model_question": question_ls,
             "mode_answer": answer_ls,
             "model_response": response_ls, 
             "ACC": acc_ls}
        )
        os.makedirs(args.save_result_dir, exist_ok=True)
        output_df.to_csv(
            result_path,
            encoding="utf-8",
            index=False
        )


    print("AVERAGE ACC:%.2f " % (sum(acc_ls) / len(acc_ls) * 100))


# In[12]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[50]:


# a = pd.read_csv(os.path.join(args.save_result_dir, f"result.csv")).loc[:,["mode_answer","model_response",'ACC']]


# aa = a[a['ACC']==False]

# print(aa.iloc[388,0])
# print(aa.iloc[388,1])


# In[32]:


# chat_tokens = tokenizer.apply_chat_template([[{'content': "no", "role": "system"},{'content': "hello", 'role': 'user'}],[{'content': "hi, help", 'role': 'user'}]], tokenize=False, add_generation_prompt=True)
# chat_tokens
# ans = model.generate(chat_tokens, eos_token_id = 128009, pad_token_id = tokenizer.pad_token_id)
# tokenizer.decode(ans, skip_special_tokens=False)


# In[13]:


# dataset['test']['question'][:2]


# In[12]:


# dataset


# In[61]:


# aaa = '''I'll help you solve the problem!

# Job A:
# Nick earns $15/hour for 2000 hours/year, so his gross income is:
# $15/hour × 2000 hours/year = $30,000/year

# After deducting 20% taxes, his take-home pay would be:
# $30,000/year × (1 - 0.20) = $24,000/year

# Job B:
# Gross income is fixed at $42,000/year. Subtracting $6000 property tax, we get:
# $42,000/year - $6000 = $36,000/year

# Net income after paying 10% tax rate on the remaining amount:
# $36,000/year × (1 - 0.10) = $32,400/year

# Now, let's compare the take-home pay of both jobs:
# Job A: $24,000/year
# Job B: $32,400/year

# To find out how much more Nick will earn at Job B, subtract the take-home pay of Job A from Job B:
# $32,400/year - $24,000/year = $8400/year

# Therefore, Nick will make $8400 more at Job B compared to Job A.

# Answer: The answer is 
# '''
# extract_answer_item(aaa)

