import os
import asyncio
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv, find_dotenv
from ollama import AsyncClient
from google import genai
from google.genai import types

load_dotenv(find_dotenv())

SYSTEM_PROMPT = """
You are a helpful assistant. I will provide a movie review and you will classify it as either positive or negative.
Please respond with "positive" or "negative" only.
"""


def trigger_openai(prompt: str):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000
    )
    response = (
        response.choices[0].message.content.strip()
        if response.choices[0].message.content
        else ""
    )
    return response


def trigger_claude(prompt: str):
    client = Anthropic()

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

def trigger_gemini(prompt: str):

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT),
        contents=prompt
    )
    return response.text


async def trigger_local_llm(prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},{'role': 'user', 'content': prompt}]
    response = await AsyncClient().chat(model='granite3.2-vision:latest', messages=messages)
    return response['message']['content']

print(trigger_gemini('Why is the sky blue?'))