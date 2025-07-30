# -*- coding: utf-8 -*-
import httpx
import ollama
from openai import OpenAI

# 配置 API 地址和模型
# api_base = 'http://localhost:11434/v1'
# model = 'nomic-embed-text'
# api_key = 'ollama'

# 定义测试文本
test_text = "Who was born earlier, Michele Padovano or Dougie Payne?"



base_url = "http://127.0.0.1:11434/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
res = client.embeddings.create(
    model="nomic-embed-text",
    input=test_text,
    ).data[0].embedding
# print("analysis:",analysis)
# score = res
print("res:",res)
# coding=utf-8
# messages = [
#         {"role": "user", "content": test_text},
#     ]
# embedding = ollama.chat(
#                     model="wen2.5:7b-instruct",
#                     messages=messages,
#                 )
# print(embedding)