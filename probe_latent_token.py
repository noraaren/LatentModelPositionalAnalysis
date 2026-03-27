#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.nn import functional as F
import json

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from peft import PeftModel
from datasets import load_dataset
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

do_print = True
probe_topk = 5
probe_idx = None
test_attention = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def parse_correct_indices(filepath):
    """Parse a decoded_latent.txt style file and return set of correct question indices."""
    correct = set()
    wrong = set()
    with open(filepath, "r") as f:
        content = f.read()
    blocks = re.split(r'\n{3,}', content)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        idx_match = re.search(r'Question(\d+)\.\.\.', block)
        if not idx_match:
            continue
        idx = int(idx_match.group(1))
        correct_match = re.search(r'Correct=(True|False)', block)
        if correct_match:
            if correct_match.group(1) == 'True':
                correct.add(idx)
            else:
                wrong.add(idx)
    return correct, wrong


def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "zen-E/GSM8k-Aug" in data_args.data_name:
        dataset = load_dataset(data_args.data_name)
        test_set = dataset['test']
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = []
    answer = []
    procedures = []
    original_indices = []  # track original dataset indices

    # Build questions with n-1 explicit CoT steps prepended
    for idx, example in enumerate(test_set):
        raw_q = example[question_name].strip().replace('  ', ' ')
        raw_cot = example["cot"]

        # Parse CoT steps e.g. <<3*3=9>>
        cot_steps = re.findall(r'<<[^>]*>>', raw_cot)
        n = len(cot_steps)

        # Prepend n-1 explicit steps to the question
        if n > 1:
            explicit_prefix = " ".join(cot_steps[:n - 1])
            q = f"{raw_q} {explicit_prefix}"
        else:
            q = raw_q

        question.append(q)
        answer.append(float(example[answer_name].replace(",", "")))
        procedures.append(raw_cot)
        original_indices.append(idx)

    # -------------------------------------------------------
    # Filter to target subset if filter files are provided
    # target subset = correct under pure latent AND wrong under mixed reasoning
    # Pass --pure_latent_file and --mixed_file as args if filtering is desired
    # -------------------------------------------------------


    #original_indices.append(idx)

    # Filter to target subset
    target_indices = {4, 20, 33, 34, 35, 58, 140, 243, 279, 319, 337, 342, 350, 354, 373, 386, 387, 396, 397, 407, 408, 420, 426, 431, 483, 508, 522, 559, 573, 577, 585, 614, 624, 643, 684, 690, 691, 700, 722, 737, 798, 803, 809, 857, 859, 870, 872, 884, 890, 893, 913, 918, 954, 982, 995, 1002, 1007, 1043, 1076, 1082, 1135, 1141, 1155, 1179, 1180, 1183, 1195, 1199, 1201, 1236, 1239, 1243, 1265, 1268, 1273, 1290, 1316}
    question       = [question[i]       for i in range(len(question))   if original_indices[i] in target_indices]
    answer         = [answer[i]         for i in range(len(answer))     if original_indices[i] in target_indices]
    procedures     = [procedures[i]     for i in range(len(procedures)) if original_indices[i] in target_indices]
    original_indices = [i               for i in original_indices       if i in target_indices]
    logging.warning(f"Filtered to target subset: {len(question)} questions")

    logging.warning("Tokenizing inputs...")

    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size} | "
                    f"eval steps: {eval_step}")

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i * data_args.batch_size: (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i * data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )

        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        # Fix: store input_len AFTER moving to device
        input_len = len(batch["input_ids"][0])
        question_data.append(batch.to(device))
        question_data[-1]['input_len'] = input_len

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []
    top5_indices_list_decoded = []
    log_count = 0
    log = []

    model.eval()
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)
        top5_values_list, top5_indices_list = [], []
        with torch.no_grad():
            # encode the question
            past_key_values = None
            outputs = model.codi(input_ids=batch["input_ids"], use_cache=True, output_hidden_states=True,
                                 past_key_values=past_key_values, attention_mask=batch["attention_mask"])
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
            top5_values_list.append(top5_values)
            top5_indices_list.append(top5_indices)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Iterate the latent thoughts
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True,
                                     past_key_values=past_key_values)
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                top5_values, top5_indices = torch.topk(probs, k=probe_topk, dim=2)
                top5_values_list.append(top5_values)
                top5_indices_list.append(top5_indices)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')).unsqueeze(0).to(device)

            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)
            output = eot_emb

            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]
            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                global_idx = step * data_args.batch_size + mini_step
                if do_print:
                    print(f"Question {global_idx} Starts...")
                    print(f"Q: {question[global_idx]}")
                    print(decoded_pred)
                    print(f"Question {global_idx} Ends")
                    print(f"Prediction={extract_answer_number(decoded_pred)}; Groundtruth={answer[global_idx]}")
                    print("")
                ans_pred_list.append(extract_answer_number(decoded_pred))

            top5_values_list = torch.cat(top5_values_list, dim=1)
            top5_indices_list = torch.cat(top5_indices_list, dim=1)

            if probe_idx is not None:
                top5_values_list = top5_values_list[:, probe_idx].unsqueeze(1)
                top5_indices_list = top5_indices_list[:, probe_idx].unsqueeze(1)

            # Log every question with Correct=True/False
            for ii in range(batch_size):
              pred = extract_answer_number(tokenizer.decode(pred_tokens[ii]))
              correct = int(answer[log_count]) == int(pred)

              log.append(f"Question{original_indices[log_count]}...")
              log.append(f"Correct={correct}")
              log.append(f"{question[log_count]}...")
              log.append(f"CoT={procedures[log_count]}, Answer={answer[log_count]}")

              for jj in range(top5_indices_list.size(1)):
                  log.append(f"decoded {jj}th latent (top5): {[tokenizer.decode(x) for x in top5_indices_list[ii, jj]]}")

              log.append(f"Model Prediction: {tokenizer.decode(pred_tokens[ii])}")
              log.append("\n\n")

              log_count += 1

    accuracy = compute_accuracy(answer, ans_pred_list)

    with open("outputs/decoded_latent.txt", "w") as f:
        f.write("\n".join(log))

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100 * accuracy:.2f}% | ")
    print(f"average length of COT: {sum(len_cot) / len(len_cot)}")

    return 100 * accuracy


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if g in p:
                acc += 1
        else:
            if p == g:
                acc += 1
    return acc / len(gold)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    for i in range(training_args.inf_num_iterations):
        accu = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
    print(f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list) / len(accu_list)}")