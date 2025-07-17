import colorsys
import os
import random
import uuid

import openai
from dotenv import load_dotenv

_client: openai.OpenAI | None = None


def random_color() -> str:
    hue = random.random()
    sat = 0.1
    val = 0.99
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def random_id() -> str:
    return str(uuid.uuid4())


def get_openai_client() -> openai.OpenAI:
    global _client

    if _client is not None:
        return _client

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")

    _client = openai.OpenAI(api_key=api_key)
    return _client
