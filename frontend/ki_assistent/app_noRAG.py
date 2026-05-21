import json
import httpx
import chainlit as cl
from chainlit.input_widget import Slider, Select, Switch, TextInput, Tags

URL = "http://localhost:1234/v1/chat/completions"
MAX_HISTORY = 20

DEFAULT_SETTINGS = {
    "model": "google/gemma-4-e2b",
    "rag_source": "Local docs",
    "temperature": 0.7,
    "rag_enabled": True,
    "system_prompt": "You are a helpful assistant.",
    "labels": ["Rat", "Beschlussvorlagen"],
}


async def show_rag_status():
    settings = cl.user_session.get("settings") or DEFAULT_SETTINGS

    rag = settings.get("rag_enabled", False)
    source = settings.get("rag_source", "None")

    status = "🟢 Abrufsystem" if rag else "⚪ Abrufsystem"

    await cl.Message(
        author='KI',
        content=f"**{status}**\n\nQuelle: `{source}`"
    ).send()


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=[
                    "openai/gpt-oss-20b",
                    "google/gemma-4-e2b",
                    "deepseek/deepseek-r1-0528-qwen3-8b",
                    "qwen/qwen3-coder-30b",
                ],
                initial_index=1,
            ),

            Slider(
                id="temperature",
                label="Temperatur",
                initial=0.7,
                min=0,
                max=2,
                step=0.1,
            ),

            Switch(
                id="rag_enabled",
                label="RAT Duisburg Retrieval",
                initial=True,
            ),

            Select(
                id="rag_source",
                label="Wissensquelle",
                values=["Ratsinformationssystem Duisburg", "None"],
                initial_index=0,
            ),

            TextInput(
                id="system_prompt",
                label="System Prompt",
                initial="You are a helpful assistant.",
                multiline=True,
            ),
            Tags(
                id="labels",
                label="Dokument-Labels",
                initial=["Rat", "Beschlussvorlagen"],
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)

    await cl.Message(author='KI', content="Settings panel ready.").send()
    await show_rag_status()


@cl.on_settings_update
async def update_settings(new_settings):
    cl.user_session.set("settings", new_settings)
    await show_rag_status()

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("settings") or DEFAULT_SETTINGS

    history = cl.user_session.get("history") or []

    history.append({
        "role": "user",
        "content": message.content
    })

    history = history[-MAX_HISTORY:]

    messages = [
        {
            "role": "system",
            "content": settings.get("system_prompt", "You are a helpful assistant."),
        },
        *history
    ]

    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": settings["temperature"],
        "stream": True,
    }

    msg = cl.Message(author='KI', content="")
    await msg.send()

    assistant_answer = ""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", URL, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data = line.removeprefix("data: ")

                if data == "[DONE]":
                    break

                chunk = json.loads(data)

                token = (
                    chunk["choices"][0]
                    .get("delta", {})
                    .get("content")
                )

                if token:
                    assistant_answer += token
                    await msg.stream_token(token)

    await msg.update()

    history.append({
        "role": "assistant",
        "content": assistant_answer
    })

    cl.user_session.set("history", history[-MAX_HISTORY:])