import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from .api import chat_with_model

task2cand_dict = {
    "sst2": ["Positive", "Negative"],
}


class ICL:
    def __init__(
        self,
        instruction,
        demonstration_list,
        test_list,
        gold_list,
        task,
        model,
        tokenizer,
        debiasing_method,
        reordering_metric,
    ):
        self.instruction = instruction
        self.demonstration_list = demonstration_list
        self.test_list = test_list
        self.gold_list = gold_list
        self.task = task
        self.cand_list = task2cand_dict[task]
        self.model = model
        self.tokenizer = tokenizer
        self.debiasing_method = debiasing_method
        self.reordering_metric = reordering_metric

    def inference_single(self, prompt, gold_truth):
        if self.debiasing_method == "reordering":
            # find_first_token_probability
            gold_token_id = self.tokenizer(gold_truth, return_tensors="pt")[
                "input_ids"
            ][0, 0]
            cand_token_id_list = [
                self.tokenizer(cand_str, return_tensors="pt")["input_ids"][0, 0]
                for cand_str in self.cand_list
            ]

            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            first_token_logits = logits[:, 0, :]

            first_token_probability = F.softmax(first_token_logits, dim=-1)

            if self.reordering_metric == "gold_prob":
                return first_token_probability[0, gold_token_id].item()
            elif self.reordering_metric == "cand_prob":
                return sum(
                    first_token_probability[0, cand_token_id].item()
                    for cand_token_id in cand_token_id_list
                )
            elif self.reordering_metric == "entropy":
                entropy = -torch.sum(
                    F.softmax(first_token_logits, dim=-1)
                    * F.log_softmax(first_token_logits, dim=-1),
                    dim=-1,
                )
                return entropy[0].item()
            else:
                raise ValueError(
                    "Invalid metric specified. Choose from 'gold_prob', 'cand_prob', or 'entropy'."
                )
        elif self.debiasing_method in ["self_exp", "baseline"]:
            return chat_with_model(self.model, prompt)
        else:
            raise ValueError(
                "Invalid debiasing method specified. Choose from 'reordering', 'self_exp', or 'baseline'"
            )

    def combine_demonstration(self, demonstration_list):
        demonstration_str = ""
        for id, demonstration in enumerate(demonstration_list):
            dm_input, dm_gold = demonstration["Input"], demonstration["Output"]
            demonstration_str += f"#{id + 1}\nInput: {dm_input}\nOutput: {dm_gold}\n"
        return demonstration_str

    def demonstration_reordering(self, reordering_metric, top_k=5):
        top_k_list = [[]]
        reordering_len = len(self.demonstration_list)
        mask_list = [False for i in range(reordering_len)]
        for search_id in range(reordering_len):
            score_list = []

            # beam search
            for tp_list in top_k_list:
                for demonstration_id in range(reordering_len):
                    if mask_list[demonstration_id] == True:
                        continue
                    mask_list[demonstration_id] = True
                    new_demonstration_list = tp_list.copy()
                    new_demonstration_list.append(
                        self.demonstration_list[demonstration_id]
                    )
                    new_demonstration_str = self.combine_demonstration(
                        new_demonstration_list
                    )

                    metric_list = []
                    for test_case in self.test_list:
                        test_input, gold_truth = test_case["Input"], test_case["Output"]
                        prompt = f"{self.instruction}\n\n{new_demonstration_str}\n\n{test_input}"
                        metric_list.append(self.inference_single(prompt, gold_truth))
                    score_list.append((np.sum(metric_list), new_demonstration_list))
            if reordering_metric == "entropy":
                top_k_id = sorted(score_list, key=lambda x: x[0])[:top_k]
            else:
                top_k_id = sorted(score_list, key=lambda x: x[0], reverse=True)[:top_k]
            top_k_list = [item[1] for item in top_k_id]
        return top_k_list

    def inference(self, save_path):
        """
        使用提供的模型和输入数据进行推理
        :return: 模型的推理结果
        """
        if self.debiasing_method in ["self_exp", "baseline"]:
            # response_list = []
            with open(save_path, "a") as fp:
                for test_case in self.test_list:
                    test_input, gold_truth = test_case["Input"], test_case["Output"]
                    prompt = f"{self.instruction}\n\n{self.combine_demonstration(self.demonstration_list)}\n\n{test_input}"
                    response = self.inference_single(prompt, gold_truth)

                    fp.write(json.dumps(response, ensure_ascii=False) + "\n")
                    # response_list.append(response)
        else:
            raise ValueError(
                "Invalid debiasing method specified. Choose from 'reordering', 'self_exp', or 'baseline'"
            )
