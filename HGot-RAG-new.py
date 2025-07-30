import asyncio
import csv
import itertools
import json
import os
import re
from time import sleep
from typing import List

import numpy as np
import openai
import pandas as pd
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.similarities import SparseMatrixSimilarity
from neo4j import GraphDatabase
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

import search
from refined import vertex_evaluator
from search import run_local_search
from tomach_kg import process_question_kg, call_embedding_api, load_entities, process_question_kg, match_entities

ollama_model = 'qwen2.5:7b-instruct'
# ollama_model = 'llama3-chatqa'


def chat_35(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ])
    return completion.choices[0].message.content


def chat_4(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ])
    return completion.choices[0].message.content


def chat_ollama(prompt):
    openai.base_url = 'http://localhost:11434/v1/'
    completion = client.chat.completions.create(
        model=ollama_model,
        # model="llama3:8b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        seed=123,
        temperature=0,
        stream=False,
        top_p=0,
        n=1)
    return completion.choices[0].message.content


def all_answers(question, qa_triples):
    template ="""Please answer the questions according to the following instructions:

                    - Identify the type of question (geographical location, historical event, person's date of birth, etc.).
                    - Extract key information from the provided triples.
                    - Use logical reasoning to determine the correct answer.
                    - Output the answer in a concise and accurate format (such as a date, place name).
                    
                    Question Example:
                    1. Are Sar Gaz, Esfandaqeh, and Do Tappeh-Ye Sofla located in the same country?
                       Related Triples: Sar Gaz - country: Iran, Esfandaqeh - country: Iran, Do Tappeh-Ye Sofla - country: Iran
                       Answer: yes
                    
                    Now, please answer the following question based on the provided triples:
                                    
                                    Question: {}
                                    Triples: {}
                                    
                                    Please provide your answer.Ensure that your response is concise and limited to a single entity.
                    """
    messages = [
        {"role": "system",
         # "content": "Based on the following triples: {}, please answer the following question: {}. If the question starts with 'which,' select the correct option from the two provided. If the question starts with 'are,' respond with 'yes' or 'no.' If the question begins with 'where,' provide the name of a place. Ensure that your response is concise and limited to the name of a single entity.".format(
         #     qa_triples, question)},
        "content":template.format(question, qa_triples)},
        # {"role": "user", "content": template2.format(question, qa_triples)}
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        seed=123,
        stream=False,
        top_p=0,
        n=1
    )
    all_answer = res.choices[0].message.content
    return all_answer



# 寻找尾实体
def find_tail_entity(head_entity, relation):
    # print("relation:", relation)
    # relation=match_entities("./2wiki/relation2id.txt", "./2wiki/re_embeddings.json", relation)
    # print("relation-new:", relation)
    with driver.session() as session:
        # head_entity=""".*{head_entity}.*""".format(head_entity)
        query = f"""
        MATCH (head:Entity)-[:{relation}]->(tail:Entity)
        WHERE head.name='{head_entity}'
        RETURN collect(tail.name) AS tail_entity
        """

        # 执行查询
        result = session.run(query)
        # 执行查询
        # result = session.run(query)
        for record in result:
            tail_entities = record["tail_entity"]
        return tail_entities


# 合并 lists
def combine_lists(*lists):
    if lists != ():
        combinations = list(itertools.product(*lists))
        results = []
        for combination in combinations:
            new_combination = []
            for sublist in combination:
                if isinstance(sublist, list):
                    new_combination += sublist
                else:
                    new_combination.append(sublist)
            results.append(new_combination)
        return results
    else:
        return []


# 获取实体的邻居
def get_entity_neighbors(entity_name: str) -> List[List[str]]:
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        neighbors = record["neighbor_entities"]

        neighbor_list.append([entity_name.replace("_", " "), rel_type.replace("_", " "),
                              ','.join([x.replace("_", " ") for x in neighbors])
                              ])

    return neighbor_list


# 构建prompt，使用知识图谱信息，创建基于路径的证据
def prompt_path_finding(path_input):
    if len(path_input) != 0:
        template = """
        There are some knowledge graph path. They follow entity->relationship->entity 
        \n\n
        {}
        \n\n
        Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...
        \n\n
    
        Output:
        """

        # 格式化模板字符串
        formatted_template = template.format(path_input)

        # 构建消息列表
        messages = [
            {"role": "system", "content": formatted_template},
            {"role": "user", "content": "{text}"}
        ]
        res = client.chat.completions.create(
            model=ollama_model,
            # model="llama3:8b",
            seed=123,
            messages=messages,
            temperature=0,
            stream=False,
            top_p=0,
            n=1)
        response_of_KG_path = res.choices[0].message.content
        return response_of_KG_path
    else:
        return []


# 构建prompt，使用知识图谱信息，创建基于邻居的证据
def prompt_neighbor(neighbor):
    if len(neighbor) != 0:
        template = """
        There are some knowledge graph. They follow entity->relationship->entity list format.
        \n\n
        {}
        \n\n
        Use the knowledge graph information. Try to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...
        \n\n
    
        Output:
        """

        # 格式化模板字符串
        formatted_template = template.format(neighbor)

        human_template = "{text}"

        # 组合系统消息和人类消息
        chat_prompt = "{}\n\n{}".format(neighbor, {})

        # 2.直接构建messages
        messages = [
            {"role": "system", "content": chat_prompt},
            # {"role": "user", "content": human_message_prompt}
            # SystemMessage(content=prompt.format(Path=path_input)),
            # HumanMessage(content="{text}")  # 这里可能需要替换为实际的文本内容
        ]
        # response_of_KG_neighbor = chat(chat_prompt_with_values.to_messages()).content
        res = client.chat.completions.create(
            model=ollama_model,
            # model="llama3:8b",
            seed=123,
            messages=messages,
            temperature=0,
            stream=False,
            top_p=0,
            n=1)
        response_of_KG_neighbor = res.choices[0].message.content
        return response_of_KG_neighbor
    else:
        return []


# 判断是否无法回答
def answer_refine(root_path, question, finalanswer):
    template = """
    Does the current answer fit the question? Answer yes or no
    Question: {}
    Answer: {}
    \n\n
    Output:
    """
    messages = [
        {"role": "user", "content": template.format(question, finalanswer)},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        seed=123,
        messages=messages,
        max_tokens=1,
        temperature=0,
    )
    if res == "yes":
        return finalanswer
    else:
        return final_search(root_path, sub_question)


# 文本处理
def autowrap_text(text, font, max_width):
    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines


# 文档匹配提示词
# def prompt_document(question, instruction):
#     template = """
#     You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
#     Patient input:\n, {}\n\n
#     You have some medical knowledge information in the following:{}
#     \n\n
#     What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
#     """.format(question, instruction)
#     messages = [
#         {"role": "system", "content": template},
#         {"role": "user", "content": "{text}"},
#     ]
#     # response_document_bm25 = chat(chat_prompt_with_values.to_messages()).content
#     res = client.chat.completions.create(
#         model=ollama_model,
#         # model="llama3:8b",
#         seed=123,
#         messages=messages,
#         temperature=0,
#         stream=False,
#         top_p=1,
#         n=1)
#     response_document_bm25 = res.choices[0].message.content
#
#     return response_document_bm25


def run_graphrag_query(root_path, query):
    search.OUTPUT_DIR = root_path
    return asyncio.run(run_local_search(query))


def document_search(root_path, sub_question, tail_entity, neighbor_list):
    if tail_entity != None and neighbor_list != None:
        # 构建提示文本模板
        prompt_template = (
            """
            You will be given an incomplete knowledge graph triple (Head Entity, Relation, ?), a potential correct Tail Entity, and some related neighbor triples. Please verify the provided Tail Entity by checking the document excerpts. If it is correct, confirm the Tail Entity and validate the provided related neighbor triples; if it is incorrect, provide the correct Tail Entity and find any relevant neighbor triples in the document. When no document excerpt is provided to verify or correct this information, rely on the model's own knowledge.
            Input:
            
            Incomplete Triple: {}
            Potential Tail Entity: {}
            Related Neighbor Triples: {}
            Your response should be formatted as follows,and Output only the result, not the title:
            Output:
            
            Confirmed/Corrected Tail Entity: 
            Validated Related Neighbor Triples: 
            For example, if the given triple is (Beijing, capital_of, ?), the potential Tail Entity is China, and the related triples are (China, capital, Beijing), (Beijing, located_in_province, Hebei Province), and the document mentions Beijing is the capital of China, then the correct response would be:
            Input:
            
            Incomplete Triple: (Beijing, capital_of, ?)
            Potential Tail Entity:  China
            Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            Output:
            
            Confirmed/Corrected Tail Entity:  China
            Validated Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            If the provided Tail Entity is incorrect, adjust your answer accordingly using the same format.
            """
        )
        prompt = prompt_template.format(sub_question, tail_entity, neighbor_list)
    if tail_entity != None and neighbor_list == None:
        # 构建提示文本模板
        prompt_template = (
            """
            You will be given an incomplete knowledge graph triple (Head Entity, Relation, ?) and a potential correct Tail Entity. Please verify the provided Tail Entity by checking the document excerpts. If it is correct, confirm the Tail Entity; if it is incorrect, provide the correct Tail Entity. Additionally, find any relevant neighbor triples in the document.When no document excerpt is provided to verify or correct this information, rely on the model's own knowledge.
            Input:
            Incomplete Triple: {}
            Potential Tail Entity: {}
            Your response should be formatted as follows,and Output only the result, not the title:
            Output:
            
            Confirmed/Corrected Tail Entity: 
            Related Neighbor Triples: 
            For example, if the given triple is (Beijing, capital_of, ?), and the Potential Tail Entity is China', and the document mentions Beijing is the capital of China.the correct response would be:
            Input:  
            Incomplete Triple: (Beijing, capital_of, ?)
            Potential Tail Entity: China
            Output:
            
            Confirmed/Corrected Tail Entity: China
            Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            If the provided Tail Entity is incorrect, adjust your answer accordingly using the same format.
            """)
        prompt = prompt_template.format(sub_question, tail_entity)
    if tail_entity == None and neighbor_list != None:
        prompt_template = (
            """
            You will be given an incomplete knowledge graph triple (Head Entity, Relation, ?) and some related neighbor triples. Please use the document excerpts to verify and complete the missing Tail Entity. If the Tail Entity can be confirmed based on the provided related triples and the document content, confirm it; otherwise, provide the correct Tail Entity. Additionally, verify the correctness of the provided related triples or provide any other relevant triples found in the document.When no document excerpt is provided to verify or correct this information, rely on the model's own knowledge.

            Input:
            
            Incomplete Triple: {}
            Related Neighbor Triples: {}
            Your response should be formatted as follows,and Output only the result, not the title:
            Output:
            
            Confirmed/Corrected Tail Entity: 
            Validated Related Neighbor Triples: 
            For example, if the given triple is (Beijing, capital_of, ?), and the related triples are ( China, capital, Beijing), (Beijing, located_in_province, Hebei Province), and the document mentions Beijing is the capital of China, then the correct response would be:
            Input:
            
            Incomplete Triple: (Beijing, capital_of, ?)
            Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            Output:
            
            Confirmed/Corrected Tail Entity: China
            Validated Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            If the provided Tail Entity or related triples are incorrect, adjust your answer accordingly using the same format.
            """
        )
        prompt = prompt_template.format(sub_question, neighbor_list)
    if tail_entity == None and neighbor_list == None:
        # 构建提示文本模板
        prompt_template = (
            """
            You will be given an incomplete knowledge graph triple (Head Entity, Relation, ?). Please use the document excerpts to verify and complete the missing Tail Entity. If the Tail Entity can be confirmed based on the document content, confirm it; otherwise, provide the correct Tail Entity. Additionally, find any relevant neighbor triples in the document.When no document excerpt is provided to verify or correct this information, rely on the model's own knowledge.

            Input:
            
            Incomplete Triple: {}
            Your response should be formatted as follows,and Output only the result, not the title:
            Output:
            
            Confirmed/Corrected Tail Entity: 
            Related Neighbor Triples: 
            For example, if the given triple is (Beijing, capital_of, ?), and the document mentions Beijing is the capital of China, then the correct response would be:
            Input:
            
            Incomplete Triple: (Beijing, capital_of, ?)
            Output:
            
            Confirmed/Corrected Tail Entity: China
            Related Neighbor Triples: (China, capital, Beijing), (Beijing, located_in_province, Hebei Province)
            If the Tail Entity cannot be directly derived from the provided information, adjust your answer accordingly using the same format.
            """
        )
        prompt = prompt_template.format(sub_question)
    response = run_graphrag_query(root_path, prompt)
    return response.response


def final_search(root_path, question):
    # 构建提示文本模板
    prompt_template = (
        # "Please answer the following question using the provided knowledge graph triples. Ensure that your response is based on the information given in these triples.\n\n"

        """First, identify the main topic of the question about {question}. Then, summarize the answer in one word or a very short phrase without additional explanation or context."""
    )
    # 使用 format 方法插入问题和三元组
    prompt = prompt_template.format(question=question)
    response = run_graphrag_query(root_path, prompt)
    return response.response


def update_sub_question(sub_question, Dependencies, Steps, Purpose, sub_qa_list):
    template = """
    Update the current sub_question: {} with the Q&A information corresponding to sub_qa_list found according to Dependencies. Only the updated sub_question is returned    """
    messages = [
        {"role": "system", "content": template.format(sub_question)},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0)
    res = res.choices[0].message.content
    print("res:", res)
    return res


def generated_sub_answer(sub_question, tail_entity, documents):
    messages = [
        {"role": "system",
         "content": "Given the incomplete triplet :{}, the possible answer:{}, and relevant document descriptions:{}, please output only the predicted tail entity.".format(
             sub_question, tail_entity, documents), },
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        n=1)
    return res.choices[0].message.content


if __name__ == "__main__":
    # 设置OpenAI API密钥
    YOUR_OPENAI_KEY = 'sk-1111111111111111111'  # replace this to your key
    root_path = 'F:/code/HToG-RAG/2wiki'
    os.environ['OPENAI_API_KEY'] = YOUR_OPENAI_KEY
    openai.api_key = YOUR_OPENAI_KEY
    # 建立到Neo4j数据库的连接
    # 1. build neo4j knowledge graph datasets构建neo4j知识图谱数据集
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()

    # 1. Neo4j build KG
    # clean all
    # session.run("MATCH (n) DETACH DELETE n")
    # read triples
    # df = pd.read_csv('./download/raw_data/2wikimultihopqa/train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'],quoting=3)
    # # 确保所有列都转换为字符串类型
    # df['head'] = df['head'].astype(str)
    # df['relation'] = df['relation'].astype(str)
    # df['tail'] = df['tail'].astype(str)
    # for index, row in df.iterrows():
    #     head_name = row['head']
    #     tail_name = row['tail']
    #     relation_name = row['relation']
    #
    #     query = (
    #             "MERGE (h:Entity { name: $head_name }) "
    #             "MERGE (t:Entity { name: $tail_name }) "
    #             "MERGE (h)-[r:`" + relation_name + "`]->(t)"
    #     )
    #     session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)

    # 2. 关键词提取和实体匹配

    OPENAI_API_KEY = YOUR_OPENAI_KEY
    base_url = "http://127.0.0.1:11434/v1/"
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    # 读取问题数据
    # with open("./download/raw_data/2wikimultihopqa/questions_and_answers_1_100.json", "r",encoding='utf-8') as f:
    with open("2wiki/qaes.json", "r", encoding='utf-8') as f:
        content = f.read()
        # 将 JSON 字符串解析为 Python 对象
        data = json.loads(content)
        # 初始化 question_list 和 output_text
        qa_triples = []
        result = []
        # 遍历 JSON 数据
        i = 1
        for item in data:
            if i <= 10:
                i = i + 1
            else:
                break

            # print("i:",i)
            # 将新的JSON数据写入文件
            with open('2wiki/reslt_subqa.json', 'w') as sbqa:
                question = item['question']
                answer = item['answer']
                print("answer:", answer)
                sub_q = item['sub_questions']
                sub_qa_list = []
                sub_answer_list = []
                all_sub_answers = []
                sub_answers = []
                for sq in sub_q:
                    # 初始化空列表
                    try:
                        sq_num = next(key for key in sq if key.startswith("Sub-question"))
                        # print("sq_num:", sq_num)
                        sub_question = sq[sq_num]
                        # print("     sub_question:", sub_question)
                    except StopIteration:
                        print("未找到以 'Sub-question' 开头的键")
                    except Exception as e:
                        print(f"发生其他错误: {e}")
                    Steps = sq["Steps to Answer"]
                    if len(sq["match_kg"]) != 0:
                        if sq["Dependencies"] != "None":
                            # 根据依赖关系，更新当前子问题
                            sub_question = sub_question.replace(sq_num, sub_qa_list[sq_num])
                            head_entity, relation = sq["match_kg"]
                            head_entity= sub_qa_list[sq_num]
                        else:
                            head_entity, relation = sq["match_kg"]
                    else:
                        head_entity, relation = sub_question.split(",")[0].split("(")[1], sub_question.split(",")[1]
                    tail_entity = find_tail_entity(head_entity, relation)
                    print(" tail_entity:", tail_entity)
                    # 知识图谱邻居实体
                    neighbor_list = []
                    neighbors = get_entity_neighbors(head_entity)
                    neighbor_list.extend(neighbors)
                    # print(" neighbor_list:", neighbor_list)

                    # 文档检索
                    documents = document_search(root_path, sub_question, tail_entity, neighbor_list)
                    # print("     文档检索结果:", documents)
                    # 将每个子问题推理出答案
                    # 8. answer generation 子问题答案生成
                    sub_answer = generated_sub_answer(sub_question, tail_entity, documents)
                    print("     sub_answer:", sub_answer)
                    # 保存子问题的结果，即？代表的尾实体
                    sub_answer_dict = {
                        sq_num: sub_answer
                    }
                    sub_qa_list.append(sub_answer_dict)
                    # 将原三元组的？用子问题答案替换
                    # if tail_entity:
                    qa_triple = {
                        sq_num: sub_question.replace("?", sub_answer)
                    }
                    # else:
                    #     qa_triple = {
                    #         sq_num: sub_question
                    #     }
                    qa_triples.append(qa_triple)
                    sub_answers.append(qa_triples)
                    all_sub_answers.append(sub_answers)
                json.dump(all_sub_answers, sbqa, indent=4)
                finally_answer = all_answers(question, qa_triples)
                qa_dict = {
                    'question': question,
                    'answer': finally_answer,
                    'ground_truth': answer
                }
                print('HGot-RAG:', finally_answer)
                print("---------------------------")
                # # 构建新的JSON数据
                result.append(qa_dict)

        # 将新的JSON数据写入文件
        with open('2wiki/reslt.json', 'w') as file:
            json.dump(result, file, indent=4)
