import chainlit as cl
import requests, json

# from chainlit.config import config



OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "mixtral"  # any model you've pulled with Ollama
# MODEL_NAME = "gpt-oss:20b"


# @cl.on_chat_start
# async def start():
#     print("Loaded config:", config.auth)
#     await cl.Message(content="Hello!").send()


@cl.on_message
async def main(prompt: cl.Message):
    # Create an empty Chainlit message to stream into
    m = cl.Message(content="")
    await m.send()

    with requests.post(
        OLLAMA_API,
        json={
            "model": MODEL_NAME,
            "prompt": prompt.content,
            "stream": True
        },
        stream=True
    ) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                token = data.get("response", "")
                if token:
                    await m.stream_token(token)

        await m.update()

        # async with cl.Message(content="").stream() as m:
        #     for line in r.iter_lines():
        #         if line:
        #             token = line.decode("utf-8")
        #             # Ollama streams JSON lines, so you'd parse & append here
        #             m.content += token
        #             await m.update()

#
# @cl.on_message
# async def main(message: cl.Message):
#     answer = query_ollama(message.content)
#
#     await cl.Message(
#         content=f"Received: {message.content}, answer: {answer}"
#     ).send()