import requests
import json

LOCAL_MODEL_BANK = {
    "vicuna-7b": "7b-url",
    "vicuna-13b": "13b-url",
    "llama2-13b": "llama-url",
}


def get_messages(
    model, input_str, api, temperature=0.0, top_p=1.0, max_tokens=2048, debug=False
):
    # 调用 arknet 接口
    headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-xxx"}
    if type(input_str) == str:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": input_str}],
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "max_tokens": max_tokens,
            "do_sample": False,
        }
    else:
        data = {
            "model": model,
            "messages": input_str,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "max_tokens": max_tokens,
            "do_sample": False,
        }

    if "llama" in model:
        data = {
            "model": model,
            "messages": [{"role": "user", "content": input_str}],
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "max_tokens": max_tokens,
            "do_sample": False,
            "instruction": True,
        }

    try:
        reply = ""
        # 调用非流式api：v2/chat/completions
        url = api + "/v1/chat/completions"
        response = requests.post(url, headers=headers, data=json.dumps(data))
        message = json.loads(response.text)

        reply = message["choices"][0]["message"]["content"]

        if not debug:
            print(message)
            print(data)
        # print(reply)
    except Exception:
        # raise Exception
        print(f"error: {response.text}")
        print(data)
        return "Error"

    return reply


def chat_with_openai(model_name, prompt):
    pass


def chat_with_model(model, prompt, sample=False, debug=False):
    # 调用 API 接口
    if model not in LOCAL_MODEL_BANK.keys():
        response = chat_with_openai(model, prompt)
    else:
        api = LOCAL_MODEL_BANK[model]
        temperature = 0.7 if sample else 0.0
        top_p = 0.95 if sample else 1.0
        response = get_messages(
            model,
            input_str=prompt,
            api=api,
            temperature=temperature,
            top_p=top_p,
            max_tokens=128,
        )
    return response
