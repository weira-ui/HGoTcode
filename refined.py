import httpx
import os
import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from typing import List, Tuple, Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from openai import OpenAI

class OllamaClient:
    def __init__(self, host=None):
        # self.host = host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            # ChatOllama(model="qwen2.5:7b-instruct")
        self.client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:11434/v1/")
        self.emb = OllamaEmbeddings(model="nomic-embed-text")
    def query(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model="qwen2.5:7b-instruct",
            messages=prompt,
            temperature=0,
            stream=False,
            )
        res = res.choices[0].message.content
        return res

    def get_embedding(self, text: str) -> list:
        return self.emb.embed_query(text)

class VertexEvaluator:
    def __init__(self, host=None):
        self.llm_client = OllamaClient(host)

    def rel_score(self, query_vector, vertex_vector):
        # return cosine_similarity(query_vector.reshape(1, -1), vertex_vector.reshape(1, -1))[0][0]
        # 确保 query_vector 和 vertex_vector 都是 NumPy 数组
        query_vector = np.array(query_vector)
        vertex_vector = np.array(vertex_vector)

        # 计算余弦相似度
        similarity = cosine_similarity(query_vector.reshape(1, -1), vertex_vector.reshape(1, -1))

        return similarity[0][0]

    def step_fit_score(self, text, query):
        # prompt = f"Does the following text fit the answer steps for the question '{query}'? Text: {text}"
        prompt = f"Does the following text fit the answer steps for the question '{query}'? Text: {text}. Please provide a probability between 0 and 1 as a floating-point number,Generate only number and nothing else"
        messages = [
            {"role": "system", "content": prompt}
        ]
        response = self.llm_client.query(messages)
        # probability = float(response.strip())
        try:
            probability = float(response)
            return probability
        except ValueError:
            return

    def score_vertex(self,query,Steps,text, query_vector, weight_rel=0.5, weight_step=0.5):
        text_vector = self.llm_client.get_embedding(text)
        rel_score = self.rel_score(query_vector, text_vector)
        step_fit_score = self.step_fit_score(text, Steps)
        return weight_rel * rel_score + weight_step * step_fit_score

    def evaluate_text_vertex(self,query,Steps, text, query_vector):
        return self.score_vertex(query,Steps,text, query_vector)

    def optimize_graph_vertex(self, main_path: List[Tuple[str, str, str]], nei_path: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        # 处理 main_path 或 side_path 可能为空的情况
        if not main_path:
            main_path = []
        if not nei_path:
            nei_path = []

        # 使用集合提高查找效率
        main_set = set(main_path)
        optimized_path = list(main_set)
        if nei_path!=[]:
            # 合并 side_path 中不存在于 main_path 的三元组
            for subject, predicate, object_ in nei_path:
                if (subject, predicate, object_) not in main_set:
                    optimized_path.append((subject, predicate, object_))

        # 删除冗余的三元组
        optimized_path = self._remove_redundant_triples(optimized_path)

        # 排序优化路径
        # optimized_path = self._sort_path(optimized_path)

        return optimized_path

    def _remove_redundant_triples(self, path: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        # 删除重复的三元组
        return list(set(path))

    # 排序
    def _sort_path(self, path: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        return sorted(path, key=lambda x: (x[0], x[1], x[2]))

    def _remove_redundant_triples(self, path: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        # 删除重复的三元组
        unique_path = []
        seen = set()
        for triple in path:
            if triple not in seen:
                unique_path.append(triple)
                seen.add(triple)
        return unique_path


    def prune_vertices(self, vertices, threshold=0.5):
        pruned = []
        seen = set()
        for v in vertices:
            if v['score'] > threshold and hashlib.md5(v['text'].encode()).hexdigest() not in seen:
                pruned.append(v)
                seen.add(hashlib.md5(v['text'].encode()).hexdigest())
        return pruned

    def sort_and_select(self, vertices):
        sorted_vertices = sorted(vertices, key=lambda x: x['score'], reverse=True)
        return sorted_vertices[:10]  # 返回前10个顶点

    def process_vertices(self, query,Steps, text_vertices=None, main_path=None, nei_path=None):
        # 获取查询向量
        query_vector = self.llm_client.get_embedding(query)

        if text_vertices is not None:
            # 如果输入的是问题和文本顶点，则进行文本顶点优化
            # print("text_vertices:",text_vertices)
            # scores = [self.evaluate_text_vertex(query,Steps,tv, query_vector) for tv in text_vertices]
            scores = [self.evaluate_text_vertex(query,Steps,text_vertices, query_vector)]
            vertices_with_scores = [{'text': text_vertices, 'score': s} for s, tv in zip(scores, text_vertices)]
            pruned_vertices = self.prune_vertices(vertices_with_scores)
            sorted_vertices = self.sort_and_select(pruned_vertices)
            return sorted_vertices

        elif main_path is not None and nei_path is not None:
            # 如果输入的是主路径和旁支路径，则进行知识图谱顶点优化
            # print("main_path:",main_path,"nei:",nei_path)
            optimized_path = self.optimize_graph_vertex(main_path, nei_path)
            return optimized_path

        else:
            raise ValueError("Invalid input. Either provide text_vertices or both main_path and side_path.")
# 创建顶点评估器实例
vertex_evaluator = VertexEvaluator()

#
# # 示例数据
# query = "Who is the director of 'Candyman: Farewell To The Flesh'?"
# text_vertices = [
#     "Candyman: Farewell to the Flesh( also known as Candyman 2) is a 1995 American supernatural slasher film and a sequel to the 1992 film\" Candyman\", an adaptation of the Clive Barker short story\" The Forbidden\"., It stars Tony Todd, Kelly Rowan, William O' Leary, Bill Nunn, Matt Clark and Veronica Cartwright., It was directed by Bill Condon and written by Rand Ravich and Mark Kruger from a story by Barker., Barker executive produced., The sequel,, was released in 1999.",
#     "经典力学无法解释原子尺度的现象..."
# ]
# Steps="Search for the director's name associated with the film 'Candyman: Farewell To The Flesh'."
# # 知识图谱示例
# main_path = [
#     ("量子力学", "is a theory", "微观粒子行为"),
#     ("量子力学", "explains", "量子现象")
# ]
# side_path = [
#     ("量子力学", "invented by", "多位科学家"),
#     ("量子力学", "related to", "波函数")
# ]
#

#
# # 文档优化示例
# print("文档优化结果:")
# document_optimization_result = vertex_evaluator.process_vertices(query,Steps=Steps, text_vertices=text_vertices)
# print(document_optimization_result)
# #
# # # 知识图谱顶点优化示例
# print("\n知识图谱顶点优化结果:")
# graph_optimization_result = vertex_evaluator.process_vertices(query,Steps=Steps, main_path=main_path, side_path=side_path)
# print(graph_optimization_result)