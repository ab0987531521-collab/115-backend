import json
import os
import requests
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m3.1-flash"


def http_get(url):
    resp = requests.get(url, timeout=10)

    return {
        "status_code": resp.status_code,
        "body_summary": resp.text[:1000],
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "Fetch a URL with HTTP GET.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的網址",
                    }
                },
                "required": ["url"],
            },
        },
    }
]



def ask_model(messages):
    api_key = os.environ["NVIDIA_API_KEY"]

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 1,
            "top_p": 0.95,
        },
        timeout=60,
    )

    print("status:", resp.status_code)
    data = resp.json()
    print(json.dumps(data, ensure_ascii=False, indent=2))

    resp.raise_for_status()
    return data


def dispatch(name, arguments):
    if name == "http_get":
        args = json.loads(arguments)
        return http_get(args["url"])

    raise ValueError(f"unknown tool: {name}")


def main():
    messages = [
        {
            "role": "user",
            "content": "請抓取 https://example.com，並簡單介紹這個網站。",
        }
    ]

    for _ in range(5):
        data = ask_model(messages)

        message = data["choices"][0]["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")

        if not tool_calls:
            print("Final answer:")
            print(message.get("content"))
            return

        for tool_call in tool_calls:
            result = dispatch(
                tool_call["function"]["name"],
                tool_call["function"]["arguments"],
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

    print("工具呼叫次數已達上限，停止執行。")

    final_data = ask_model(messages)
    print("Final answer:")
    print(final_data["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()