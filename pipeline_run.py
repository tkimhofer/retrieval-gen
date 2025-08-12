from services.datahandler import LLMRun
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv('.env')

api_key = os.getenv('GPT_API_KEY')
client = OpenAI(api_key)


run = LLMRun(client)