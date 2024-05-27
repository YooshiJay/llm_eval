#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
from typing import List
import pandas as pd
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.generation import GenerationConfig
from datasets import load_dataset, load_from_disk


class args:
    checkpoint_path = '/gemini/code/lamma3_eval/lamma3_model/8B_instruct'
    eval_data_path = '/gemini/code/lamma3_eval/eval_data/mmlu'
    save_result_dir = "/gemini/code/lamma3_eval/eval_result/mmlu_chat"
    choices = ["A", "B", "C", "D"]
    debug = False
    overwrite = False
    batch_size = 4
    max_seq_len = 1536


# In[2]:


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]


# In[3]:


def load_models_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side='left',
        pad_token='<|reserved_special_token_0|>'
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
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path
    )
    model.generation_config.do_sample = False  # use greedy decoding
    model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    return model, tokenizer


# In[4]:


model, tokenizer = load_models_tokenizer()
# dataset = load_from_disk(args.eval_data_path)
# dev = dataset['dev']


# In[5]:


def format_example(line, include_answer=True):
    example = "Question: " + line["question"]
    for i, choice in enumerate(args.choices):
        example += f'\n{choice}. {line["choices"][i]}'

    if include_answer:
        example += "\nAnswer: " + args.choices[line["answer"]] + "\n\n"
    else:
        example += "\nAnswer:"
    return example


def generate_few_shot_prompt(dev, subject_name):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject_name))

    for i in range(len(dev)):
        prompt += format_example(
            dev[i],
            include_answer=True,
        )
    return prompt


def doc_to_text(doc, few_shot_prompt): # doc 是 dataset
    return batch_process(lambda x: few_shot_prompt + format_example(x, include_answer=False), doc)


def batch_process(func, *args):
    '''
    args 负责接受 一个或多个 batch
    '''
    
    text_ls = []
    # 只收到一个迭代元素时，zip会自动 将其的每个元素 单独包装成一个元组
    # [1, 2, 3] -> [(1,), (2,), (3,)]
    for sample in zip(*args): 
        text_ls.append(func(*sample))
    return text_ls


# In[6]:


def get_logits(tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
    # print(input_ids["input_ids"].shape[1])
    cur_len = input_ids["input_ids"].shape[1]
    if cur_len > args.max_seq_len:
        input_ids["input_ids"] = input_ids["input_ids"][:, cur_len - args.max_seq_len :]
        input_ids["attention_mask"] = input_ids["attention_mask"][:, cur_len - args.max_seq_len :]

    tokens = {"input_ids": input_ids}

    outputs = model(**input_ids)["logits"]
    logits = outputs[:, -1, :]  # (batch, 每个子段 对下个token的预测, 词表size)  只要下个token的预测，取最后一个
    # log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return logits, {"tokens": tokens}


def is_correct(pred, answer):
    return batch_process(lambda x, y: x==y, pred, answer)


# In[7]:


# gather 表示根据 index “聚集” logits 对应位置的元素
# 在 index 张量中，一个元素会处在某个位置，dim 表示：替换这个元素位置（会是张量维度大小）的第dim维度 -> 元素的值
# 比如 dim = -1, index[0,0] = 1，含义为：结果的[0,0]位置 填 input[0,1]
# input:
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14]])
# index:
# tensor([[1, 0],
#         [0, 0],
#         [1, 2]])
# dim=1时:
# tensor([[ 1,  0],
#         [ 5,  5],
#         [11, 12]])

@torch.no_grad()
def eval_subject(subject_name, dataset):
    # torch.cuda.empty_cache()
    test = dataset['test']
    dev = dataset['dev']
    question_ls = []
    answer_ls = []
    score = []
    result = []

    few_shot_prompt = generate_few_shot_prompt(dev, subject_name) # 5 shot
    if args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")
    
    choices_ids = torch.tensor(
        tokenizer(" A")["input_ids"][1:] + tokenizer(" B")["input_ids"][1:] +
        tokenizer(" C")["input_ids"][1:] + tokenizer(" D")["input_ids"][1:]
    ).unsqueeze(0).to(model.device)

    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    for i in tqdm(range(0, len(test), args.batch_size)):
        batch = test.select(range(i, min(i+args.batch_size, len(test))))
        context = doc_to_text(batch, few_shot_prompt)
        logits, input_info = get_logits(tokenizer, model, context)
        
        softval = logits.gather(dim=1, index=choices_ids.expand(logits.size(0), -1)).softmax(1)
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()
        if args.debug:
            print(probs)

        for i in range(len(probs)):
            for j, choice in enumerate(args.choices):
                all_probs[f"prob_{choice}"].append(probs[i][j])
        
        pred = np.argmax(probs, axis=-1)
        answer = batch['answer']
        acc = is_correct(pred, answer)

        if args.debug:
            for i in range(len(batch)):
                print(f'{batch["question"][i]} \npred: {pred[i]} \nref: {answer[i]}\n')
            
        question_ls.extend(context)
        answer_ls.extend(answer)
        result.extend(pred)
        score.extend(acc)
         

    return question_ls, answer_ls, result, score


# In[8]:


def main():
    all_question = []
    all_answer = []
    all_result = []
    all_score = []

    # 看有无文件，有的话就不重复做了
    result_path = os.path.join(args.save_result_dir, f"result.csv")
    if not args.overwrite and os.path.exists(result_path):
        print(f"{result_path} existed, skip!")
        for (_, resultrow) in pd.read_csv(result_path).iterrows():
            # pred = extract_answer(resultrow['model_response'], datarow)
            acc = resultrow["ACC"]
            all_score.append(acc)

    else:
        for subject_name in tqdm(SUBJECTS):
            # print(subject_name)
            dev_file_path = os.path.join(
                args.eval_data_path, subject_name, "dev-00000-of-00001.parquet"
            )
            test_file_path = os.path.join(
                args.eval_data_path, subject_name, "test-00000-of-00001.parquet"
            )

            dataset = load_dataset("parquet", data_files={'dev': dev_file_path, 'test': test_file_path})
            
            question_ls, answer_ls, result, score = eval_subject(subject_name, dataset)
            all_question.extend(question_ls)
            all_answer.extend(answer_ls)
            all_result.extend(result)
            all_score.extend(score)


        # 存入文件
        output_df = pd.DataFrame(
            {"model_question": all_question,
            "standard_answer": all_answer,
            "model_response": all_result, 
            "ACC": all_score}
        )
        os.makedirs(args.save_result_dir, exist_ok=True)
        result_path = os.path.join(args.save_result_dir, f"result.csv")
        output_df.to_csv(
            result_path,
            encoding="utf-8",
            index=False
        )

    print("AVERAGE ACC:%.2f " % (sum(all_score) / len(all_score) * 100))


# In[14]:


# !export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192

if __name__ == "__main__":
    # high_school_european_history 中 question + few_shot 达到 3000 token，需要限制token数量。
    main()


# In[ ]:





# In[ ]:




