from collections import defaultdict
import json
import random
import os


def load_demonstration(demonstration_mode, sample_id, case_num, task):
    # 读取 path 对应的 demonstration 字符串内容
    with open(
        f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}/{case_num}/{task}.json",
        "r",
    ) as fp:
        json_str = json.load(fp)
    demonstration_str = json.loads(json_str)
    return demonstration_str


def load_output(output_dir, prefix, suffix="prediction"):
    to_load_list = list()

    file_dir = os.path.join(output_dir, f"{prefix}_{suffix}.json")
    if not os.path.exists(file_dir):
        if "trec" in file_dir:
            for i in range(0, 480):
                to_load_list.append("Error")
        else:
            for i in range(0, 500):
                to_load_list.append("Error")
        return to_load_list

    with open(file_dir, "r") as fp:
        lines = fp.readlines()

    for line in lines:
        try:
            line = json.loads(line)
        except Exception:
            pass
        if suffix == "prediction":
            line = line.replace("\n", "")
        to_load_list.append(line)

    return to_load_list


def save_demonstration(
    demonstration_mode, sample_id, case_num, task, demonstration_str
):
    # 将某个 task 的 demonstration 字符串格式化存储到 path 对应的 json/jsonl 文件中
    json_str = json.dumps(demonstration_str)

    if not os.path.exists(f"./demonstrations/demonstrations_{sample_id}"):
        os.mkdir(f"./demonstrations/demonstrations_{sample_id}")
    if not os.path.exists(
        f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}"
    ):
        os.mkdir(f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}")
    if not os.path.exists(
        f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}/{case_num}"
    ):
        os.mkdir(
            f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}/{case_num}"
        )
    with open(
        f"./demonstrations/demonstrations_{sample_id}/{demonstration_mode}/{case_num}/{task}.json",
        "w",
    ) as fp:
        json.dump(json_str, fp)


def sample_cases(
    task, split, sample_mode="test", num_sample=50, sample_id=None, fq_bias=False
):
    """ """

    task2feature = {
        "sst2": [
            ["sentence"],
            "label",
            {
                1: "Positive",
                0: "Negative",
            },
            {
                "Positive": ["Negative", "Foo"],
                "Negative": ["Positive", "Bar"],
            },
        ],
        "fp": [
            ["Sentence"],
            "Sentiment",
            {
                "positive": "Positive",
                "negative": "Negative",
            },
            {
                "Positive": ["Negative", "Foo"],
                "Negative": ["Positive", "Bar"],
            },
        ],
        "ethos": [
            ["text"],
            "label",
            {
                1: "Hate Speech",
                0: "Not Hate Speech",
            },
            {
                "Hate Speech": ["Not Hate Speech", "Bar"],
                "Not Hate Speech": ["Hate Speech", "Foo"],
            },
        ],
        "trec": [
            ["text"],
            "coarse_label",
            {
                0: "Abbreviation",
                1: "Entity",
                2: "Description and Abstract Concept",
                3: "Human Being",
                4: "Location",
                5: "Numeric Value",
            },
            {},
        ],
    }

    # 随机选取 test example
    with open(f"./data/{task}/{split}.json", "r") as fp:
        raw_lines = fp.readlines()
    lines = [json.loads(raw_line) for raw_line in raw_lines]

    sample_dict = defaultdict()
    input_feature_list, output_feature, mapping_dict, flip_sul_dict = task2feature[task]
    num_class = len(mapping_dict)

    for key, val in mapping_dict.items():
        sample_dict[key] = list()
    for line in lines:
        if line[output_feature] in sample_dict.keys():
            sample_dict[line[output_feature]].append(line)

    sample_list = list()
    dict_list = defaultdict()
    for key, lst in sample_dict.items():
        dict_list[key] = list()
        for index in random.sample(range(len(lst)), num_sample):
            dict_list[key].append(lst[index])

    for i in range(num_sample):
        for key, lst in dict_list.items():
            sample_list.append(lst[i])

    formatted_list = list()
    for raw_item in sample_list:
        item = dict()
        item["Input"] = "\n".join(
            raw_item[input_feature] for input_feature in input_feature_list
        )
        item["Output"] = mapping_dict[raw_item[output_feature]]

        formatted_list.append(item)

    if sample_mode == "test":
        json_list = json.dumps(formatted_list)
        with open(f"./test_case/{task}.json", "w") as fp:
            json.dump(json_list, fp)
    elif sample_mode.startswith("demonstration"):
        # 按照格式排列
        std_str = ""
        for item in formatted_list:
            std_str += f"Input: {item['Input']}\nOutput: {item['Output']}\n"
        std_str = add_sign(std_str)
        save_demonstration("std", sample_id, num_sample, task, std_str)

        if task == "trec":
            return

        flipped_str = ""
        for item in formatted_list:
            item["Output"] = flip_sul_dict[item["Output"]][0]
            flipped_str += f"Input: {item['Input']}\nOutput: {item['Output']}\n"
        flipped_str = add_sign(flipped_str)
        save_demonstration("flipped", sample_id, num_sample, task, flipped_str)

        sul_str = ""
        for item in formatted_list:
            item["Output"] = flip_sul_dict[item["Output"]][1]
            sul_str += f"Input: {item['Input']}\nOutput: {item['Output']}\n"
        sul_str = add_sign(sul_str)
        save_demonstration("sul", sample_id, num_sample, task, sul_str)


def load_test_case(path):
    # 读取 path 对应的 test case 字符串内容
    with open(f"./test_case/{path}.json", "r") as fp:
        json_list = json.load(fp)
    sample_test_list = json.loads(json_list)
    return sample_test_list


def add_sign(demonstration_str):
    # 为每个 case 前面加标识符
    case_list = demonstration_str.split("Input: ")
    case_list = case_list[1:]

    new_str = ""
    for id, case_item in enumerate(case_list):
        new_str += f"(#{id + 1})\n"
        new_str += "Input: " + case_item
    return new_str


def main():
    # for task in ["sst2", "subj", "fp", "rte", "ethos", "qqp", "rotten_tomatoes"]:
    #     # sample_cases(task, "train", "test", 250)
    #     for sample_id in range(0, 10):
    #         sample_cases(task, "train", "demonstration", 16, sample_id)
    #         sample_cases(task, "train", "demonstration", 8, sample_id)
    #         sample_cases(task, "train", "demonstration", 4, sample_id)
    #         sample_cases(task, "train", "demonstration", 2, sample_id)
    # # sample_cases("trec", "train", "test", 80)
    # for sample_id in range(0, 10):
    #     sample_cases("trec", "train", "demonstration", 16, sample_id)
    #     sample_cases("trec", "train", "demonstration", 8, sample_id)
    #     sample_cases("trec", "train", "demonstration", 4, sample_id)
    #     sample_cases("trec", "train", "demonstration", 2, sample_id)

    dm_str = load_demonstration("std", 10, 16, "sst2")
    print(dm_str)


if __name__ == "__main__":
    main()
