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
import sys
ollama_model = 'qwen2.5:7b-instruct'
# ollama_model = 'llama3.1:8b'
sys.stdout.reconfigure(encoding='utf-8')

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


def chat_ollama(question_string):
    openai.base_url = 'http://localhost:11434/v1/'
    io_prompt = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
    A: {Washington, D.C.}.

    Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
    A: {Bharoto Bhagyo Bidhata}.

    Q: Who was the artist nominated for an award for You Drive Me Crazy?
    A: {Jason Allen Alexander}.

    Q: What person born in Siegen influenced the work of Vincent Van Gogh?
    A: {Peter Paul Rubens}.

    Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
    A: {Georgia}.

    Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
    A: {Heroin}."""
    prompt = io_prompt + "\n\nQ: " + question_string + "\nA: "
    messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."}]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)
    completion = client.chat.completions.create(
        model=ollama_model,
        # model="llama3:8b",
        messages=messages,
        seed=123,
        temperature=0,
        stream=False,
        top_p=0,
        n=1)
    return completion.choices[0].message.content


def all_answers(yn,question, sub_qa):
    if yn:
        yes_no="You only need to answer yes or no to this question"
    else:
        yes_no=""
    messages = [
        {
            "role": "system",
            "content": """will provide a complex problem and a set of subproblems and corresponding answers to the complex problem when it is broken down.First, determine the type of answer needed for the main question (e.g., yes/no, full time, location, etc.) based on the following sub-questions and their corresponding answers. Then, derive the answer step by step in the order of dependencies, summarizing it in a entity phrase without any explanation or context.Such as answer xxx instead of xxx has more floors.
                {}
                Input:
                The question you should answer:{}
                
                helpful Sub-questions and Answers:{}
                
                
                Output: Final Answer
                
            """.format(yn,question, sub_qa)

        }
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        n=1
    )
    all_answer = res.choices[0].message.content
    return all_answer


def Problem_vertex_inference(input_text):
    template = """<Instruction> Please update the answer vertex according to the answer steps for the following question and then rate it. The rating criteria are as follows:

            - 10 points: The updated answer vertex perfectly answers the question in accordance with the answer steps, without omitting critical information and without introducing irrelevant information.
            - 0 points: The updated answer vertex does not answer the question following the answer steps, or significantly omits critical information, or introduces a substantial amount of irrelevant information.
            Input:
                Question: [question]
                Answer Steps: [Answer steps]
                Graph Vertex Set: [GraphVertices]
                Text Vertex Set: [TextVertices]
            Output:
                Updated Answer Vertex: [UpdatedVertex]
                Rating: [Score] (Please provide the rating between the tags <Score> and </Score>, without including any other text.)
            </Instruction>

            <Example>
            Input:
               Question: How does global warming affect the habitat of polar bears?
               Answer Steps: 1. Identify the phenomenon of global warming; 2. Identify changes in the polar bear habitat; 3. Analyze how global warming influences the habitat of polar bears.
               Graph Vertex Set: [GlobalWarming, ArcticHabitat, IceMelting, PolarBears]
               Text Vertex Set: [TemperatureRise, AnimalDisplacement, HabitatShrinkage]
            Output:
               Updated Answer Vertex: GlobalWarmingImpactOnArcticHabitatOfPolarBears
               <Score>
               9
               </Score>
            </Example>
            Input:
            Question: {input1}
            Answer Steps: {input2}
            Graph Vertex Set: {input3}
            Text Vertex Set: {input4}
            Output:
            """
    messages = [
        {"role": "system", "content": template.format(input1=input_text[0], input2=input_text[1], input3=input_text[2],
                                                      input4=input_text[3])},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        seed=123,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        n=1)
    Problem_vertex_inference = res.choices[0].message.content


# 寻找最短路径
def find_shortest_path(start_entity_name, end_entity_name, candidate_list):
    global exist_entity
    exist_entity = {}  # 初始化全局变量
    with driver.session() as session:
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)

            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_", " ")

                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_", " ")
                    path_str += "->" + relations[i] + "->"

            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}

        if len(paths) > 5:
            paths = sorted(paths, key=len)[:5]

        return paths, exist_entity


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


def answer_refine(question, finalanswer):
    template = """
    Please determine the type of answer required for the main question (e.g., yes/no, time, place,  etc.). Then, summarize and output the answers to the following questions with one word or entity,If you can't give an answer,Output: Unknown
    question: {}
    finalanswer: {}
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
        temperature=0,
        top_p=0,
        n=1
    )
    return res.choices[0].message.content


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
#         top_p=0,
#         n=1)
#     response_document_bm25 = res.choices[0].message.content
#
#     return response_document_bm25


def run_graphrag_query(root_path, query):
    search.OUTPUT_DIR = root_path
    # search.INPUT_DIR = root_path
    return asyncio.run(run_local_search(query))


def document_search(root_path, sub_question, text, refined_kg):
    # print("refined_kg:",refined_kg)
    if refined_kg != [] and refined_kg != [[]] and refined_kg != None:
        # 构建提示文本模板
        prompt_template = (
            # "Find all the content that you think is helpful in solving the problem and related to the triples provided"
            "Find all documentation related to the problem, the triples provided, and the set of entities provided,If the information provided does not contain any direct reference, please make an inference to give the maximum possible result, while giving possible answers and do not any explanations"
            "Question:{question}"
            # "Possible related knowledge graph triples:{triples}"
            # "Related context:{text}"
        )
        prompt = prompt_template.format(question=sub_question, triples=refined_kg, text=text)
    else:
        prompt_template = (
            "Find all the content that you think is helpful in solving the problem,If the information provided does not contain any direct reference, please make an inference to give the maximum possible result, while giving possible answers and do not any explanations"
            "Question:{question}"
            "Related context:{text}"
        )
        prompt = prompt_template.format(question=sub_question, text=text)
    response = run_graphrag_query(root_path, prompt)
    # print(response.response)
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


def update_sub_question(sub_question, Dependencies, sub_qa_list):
    template = """
    Update the current sub_question: {} with the Q&A information corresponding to {} found according to {}. Only the updated sub_question is returned    """
    messages = [
        {"role": "system", "content": template.format(sub_question, sub_qa_list, Dependencies)},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0)
    res = res.choices[0].message.content
    # print("res:", res)
    return res


def generated_sub_answer(sub_qa_list, sub_question, graph_optimization_result, document):
    messages = [
        # {"role": "system",
        #  "content": "You are a knowledgeable AI assistant who can answer any type of question. Your responses should be correct and concise. You will receive the following inputs: 1.Questions and answers about possible relationships,2. The question,  3.. Knowledge graph triplets related to the main entities in the question, and 4. documents.Please provide your answer.Ensure that your response is concise and limited to a single entity."},

        {"role": "system",
         "content": """ {}
                        Below are some of the information that has been retrieved:
                        Q&a information about related content: {}
                        Related triplet information: {}
                        Reasoning information about the problem:{}
                        Analyze the available information and  give the most accurate answer""".format(sub_question,
                                                                                                       sub_qa_list,
                                                                                                       graph_optimization_result,
                                                                                                       document)},

        # {"role": "assistant",
        # "content": "Based on the provided information, here is the answer: The capital of France is Paris."}
    ]

    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        seed=123,
        n=1)
    # print("res:", res.choices[0].message.content)
    return res.choices[0].message.content


def answer_retrieve(question, text):
    io_prompt = """You can modify your question and answer according to this format. For example:
        Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
        context:[.....]
        A: Washington, D.C.

        Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
        context:[.....]
        A: Bharoto Bhagyo Bidhata

        Q: Who was the artist nominated for an award for You Drive Me Crazy?
        context:[.....]
        A: Jason Allen Alexander

        Q: What person born in Siegen influenced the work of Vincent Van Gogh?
        context:[.....]
        A: Peter Paul Rubens

        Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
        context:[.....]
        A: Georgia

        Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
        context:[.....]
        A: Heroin"""
    messages = [
        {"role": "system",
         # "content": "You are a knowledgeable AI assistant who can answer any type of question. Your responses should be correct and concise. You will receive the following inputs: 1.Question 2. Relation Context "},
         "content": "You are a knowledgeable AI assistant who can answer any type of question.You can modify your question and answer according to this format.The word and context must be the same. The time must be the whole year, month and day.For example:"+io_prompt},

        {"role": "system",
         "content": """ 
         Please determine the type of answer required for the main question (e.g., yes/no, time, place,  etc.).There is a certain similarity when judging yes or no, that is, yes, it does not need to be perfect Then, summarize and output the answers to the following questions with one word or entity
         q:{},
         context: {},
         a:
         """.format(question, text)},
    ]

    res = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        seed=123,
        n=1)
    return res.choices[0].message.content


def prompt_document(question, instruction):
    template = """
    You are an excellent AI doctor, and you can diagnose diseases and recommend medications based on the symptoms in the conversation.\n\n
    Patient input:\n, {}\n\n
    You have some medical knowledge information in the following:{}
    \n\n
    What disease does the patient have? What tests should patient take to confirm the diagnosis? What recommened medications can cure the disease?
    """.format(question, instruction)

    # prompt = PromptTemplate(
    #     template=template,
    #     input_variables=["question", "instruction"]
    # )
    #
    # system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    # system_message_prompt.format(question=question,
    #                              instruction=instruction)
    #
    # human_template = "{text}"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # chat_prompt_with_values = chat_prompt.format_prompt(question=question, \
    #                                                     instruction=instruction, \
    #                                                     text={})
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": "{text}"},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        # model="llama3:8b",
        seed=123,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=1,
        n=1)
    response_document_bm25 = res.choices[0].message.content

    return response_document_bm25


def is_unable_to_answer(response):
    # analysis = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=response,
    #     max_tokens=1,
    #     temperature=0.0,
    #     n=1,
    #     stop=None,
    #     presence_penalty=0.0,
    #     frequency_penalty=0.0
    # )
    messages = [
        {"role": "user", "content": response},
    ]
    analysis = client.chat.completions.create(
        model=ollama_model,
        # model="llama3:8b",
        seed=123,
        messages=messages,
        max_tokens=1,
        temperature=0,
        stream=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_p=1,
        n=1)
    # print("analysis:",analysis)
    score = analysis.choices[0].message.content.strip().replace("'", "").replace(".", "")
    if not score.isdigit():
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


if __name__ == "__main__":
    # 设置OpenAI API密钥
    YOUR_OPENAI_KEY = 'sk-1111111111111111111'  # replace this to your key
    # root_path = 'F:/code/HToG-RAG/2wiki'
    root_path = 'F:/code/HToG-RAG/hotpotqa'
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

    # chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # 回答记录输出
    # with open('output.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(
    #         ['Question', 'Label', 'HGoT-RAG', 'LLM', 'BM25_retrieval', 'Embedding_retrieval', 'KG_retrieval'])

    # 读取文档
    # docs_dir = './data/2wikimultihopqa/document'

    # docs = []
    # for file in os.listdir(docs_dir):
    #     with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
    #         doc = f.read()
    #         docs.append(doc)
    # 读取问题数据
    # with open("./download/raw_data/2wikimultihopqa/questions_and_answers_1_100.json", "r",encoding='utf-8') as f:
    # with open("2wiki/qaes.json", "r", encoding='utf-8') as f:
    with open("hotpotqa/qaes.json", "r", encoding='utf-8') as f:
        content = f.read()
        # 将 JSON 字符串解析为 Python 对象
        data = json.loads(content)

        # 初始化 question_list 和 output_text
        qa_list = []  # 用于子问题和答案存储问题
        qa_right_list = []
        all_sub = []  # 存储所有子问题结果
        result = []  # 只存储问题和答案
        ollama_qa=[]
        document_qa = []
        kg_qa = []
        # 遍历 JSON 数据
        i = 0
        for item in data:
        #     # if i >= 1:
        #     #     break
                i = i + 1
                print("question", i)
        #     # 将新的JSON数据写入文件
        #     with open('hotpotqa/reslt_subqa.json', 'w') as sbqa:
                question = item['question']
                answer = item['answer']
        #         # print("question",i)
        #         print("answer:", answer)
        #         sub_q = item['sub_questions']
        #         # print(sub_q)
        #         # 将 问题和正确答案 添加到 qa_right_list
        #         # 将问题进行子问题划分-解答步骤-依赖处理
        #         # sub_q = prompt_to_subquestions(question)
        #         sub_qa_list = []
        #         for sq in sub_q:
        #             # 初始化空列表
        #             try:
        #                 sq_num = next(key for key in sq if key.startswith("Sub-question"))
        #                 sub_question = sq[sq_num]
        #                 # print("     sub_question:", sub_question)
        #             except StopIteration:
        #                 print("未找到以 'Sub-question' 开头的键")
        #             except Exception as e:
        #                 print(f"发生其他错误: {e}")
        #             Steps = sq["Steps to Answer"]
        #             # Purpose = sq["Purpose"]
        #             if sq["Dependencies"] != "None":
        #                 #     根据依赖关系，更新当前子问题
        #                 sub_question = update_sub_question(sub_question, sq["Dependencies"], sub_qa_list)
        #             # question_kg = Entity_recognition(sub_question)
        #             #     # print("question_kg:", question_kg)
        #             #     # 实体对齐
        #             #     # match_kg = match_entities("./download/raw_data/2wikimultihopqa/embeddings.json",'embeddings.json',question_kg)
        #             if len(sq["match_kg"]) != 0:
        #                 match_kg = sq["match_kg"]
        #             else:
        #                 match_kg = sq["question"]
        #             #     # print("match_kg:", match_kg)
        #             #     # # 4. neo4j knowledge graph path findingNeo4j知识图谱路径查找
        #             if len(match_kg[0]) != 0:
        #                 start_entity = match_kg[0]
        #                 candidate_entity = match_kg[1:]
        #
        #                 result_path_list = []
        #                 while 1:
        #                     flag = 0
        #                     paths_list = []
        #                     while candidate_entity != []:
        #                         end_entity = candidate_entity[0]
        #                         candidate_entity.remove(end_entity)
        #                         paths, exist_entity = find_shortest_path(start_entity, end_entity, candidate_entity)
        #                         path_list = []
        #                         if paths == [''] or paths == []:
        #                             flag = 1
        #                             if candidate_entity == []:
        #                                 flag = 0
        #                                 break
        #                             start_entity = candidate_entity[0]
        #                             candidate_entity.remove(start_entity)
        #                             break
        #                         else:
        #                             for p in paths:
        #                                 path_list.append(p.split('->'))
        #                             if path_list != []:
        #                                 paths_list.append(path_list)
        #
        #                         if exist_entity != {}:
        #                             try:
        #                                 candidate_entity.remove(exist_entity)
        #                             except:
        #                                 continue
        #                         start_entity = end_entity
        #                     result_path = combine_lists(*paths_list)
        #
        #                     if result_path != []:
        #                         result_path_list.extend(result_path)
        #                     if flag == 1:
        #                         continue
        #                     else:
        #                         break
        #
        #                 start_tmp = []
        #                 for path_new in result_path_list:
        #
        #                     if path_new == []:
        #                         continue
        #                     if path_new[0] not in start_tmp:
        #                         start_tmp.append(path_new[0])
        #
        #                 if len(start_tmp) == 0:
        #                     result_path = {}
        #                     single_path = {}
        #                 else:
        #                     if len(start_tmp) == 1:
        #                         result_path = result_path_list[:5]
        #                     else:
        #                         result_path = []
        #
        #                         if len(start_tmp) >= 5:
        #                             for path_new in result_path_list:
        #                                 if path_new == []:
        #                                     continue
        #                                 if path_new[0] in start_tmp:
        #                                     result_path.append(path_new)
        #                                     start_tmp.remove(path_new[0])
        #                                 if len(result_path) == 5:
        #                                     break
        #                         else:
        #                             count = 5 // len(start_tmp)
        #                             remind = 5 % len(start_tmp)
        #                             count_tmp = 0
        #                             for path_new in result_path_list:
        #                                 if len(result_path) < 5:
        #                                     if path_new == []:
        #                                         continue
        #                                     if path_new[0] in start_tmp:
        #                                         if count_tmp < count:
        #                                             result_path.append(path_new)
        #                                             count_tmp += 1
        #                                         else:
        #                                             start_tmp.remove(path_new[0])
        #                                             count_tmp = 0
        #                                             if path_new[0] in start_tmp:
        #                                                 result_path.append(path_new)
        #                                                 count_tmp += 1
        #
        #                                         if len(start_tmp) == 1:
        #                                             count = count + remind
        #                                 else:
        #                                     break
        #
        #                     try:
        #                         single_path = result_path_list[0]
        #                     except:
        #                         single_path = result_path_list
        #
        #             else:
        #                 result_path = {}
        #                 single_path = {}
        #             #     # print("result_path:", result_path)
        #             #     print(" single_path:", single_path)
        #             #     # # 5. neo4j knowledge graph neighbor entities Neo4j知识图谱邻居实体
        #             neighbor_list = []
        #             for match_entity in match_kg:
        #                 neighbors = get_entity_neighbors(match_entity)
        #                 neighbor_list.extend(neighbors)
        #             # print(" neighbor_list:", neighbor_list)
        #             #     # 6. knowledge gragh path based prompt generation 基于知识图谱路径的提示生成
        #             #     if len(match_kg) != 1 or 0:
        #             #         response_of_KG_list_path = []
        #             #         if result_path == {}:
        #             #             response_of_KG_list_path = []
        #             #         else:
        #             #             result_new_path = []
        #             #             for total_path_i in result_path:
        #             #                 path_input = "->".join(total_path_i)
        #             #                 result_new_path.append(path_input)
        #             #
        #             #             path = "\n".join(result_new_path)
        #             #             response_of_KG_list_path = prompt_path_finding(path)
        #             #             if is_unable_to_answer(response_of_KG_list_path):
        #             #                 response_of_KG_list_path = prompt_path_finding(path)
        #             #             print("response_of_KG_list_path:", response_of_KG_list_path)
        #             #     else:
        #             #         response_of_KG_list_path = '{}'
        #             #
        #             #     response_single_path = prompt_path_finding(single_path)
        #
        #             # print("response_single_path:",response_single_path)
        #             # # 7. knowledge gragh neighbor entities based prompt generation   提示生成知识图谱近邻实体
        #             # response_of_KG_list_neighbor = []
        #             # neighbor_new_list = []
        #             # for neighbor_i in neighbor_list:
        #             #     neighbor = "->".join(neighbor_i)
        #             #     neighbor_new_list.append(neighbor)
        #             #
        #             # if len(neighbor_new_list) > 5:
        #             #     neighbor_new_list = "\n".join(neighbor_new_list[:5])
        #             # response_of_KG_neighbor = prompt_neighbor(neighbor_new_list)
        #             # if is_unable_to_answer(response_of_KG_neighbor):
        #             #     response_of_KG_neighbor = prompt_neighbor(neighbor_new_list)
        #             # print("response_of_KG_neighbor:",response_of_KG_neighbor)
        #             # bm25
        #             # document_dir = "./download/raw_data/2wikimultihopqa/document_1_100"
        #             # document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]
        #             # corpus = []
        #             # for path in document_paths:
        #             #     with open(path, "r", encoding="utf-8") as f:
        #             #         corpus.append(f.read().lower().split())
        #             #
        #             # dictionary = corpora.Dictionary(corpus)
        #             # bm25_model = BM25Okapi(corpus)
        #             #
        #             # bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
        #             # bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))
        #             #
        #             # query = input_text[0]
        #             # query_tokens = query.lower().split()
        #             # tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
        #             # tfidf_query = tfidf_model[dictionary.doc2bow(query_tokens)]
        #             # best_document_index, best_similarity = 0, 0
        #             #
        #             # bm25_scores = bm25_index[tfidf_query]
        #             # for i, score in enumerate(bm25_scores):
        #             #     if score > best_similarity:
        #             #         best_similarity = score
        #             #         best_document_index = i
        #             #
        #             # with open(document_paths[best_document_index], "r", encoding="utf-8") as f:
        #             #     best_document_content = f.read()
        #             # response_of_document = prompt_document(input_text[0], best_document_content)
        #             # print("response_of_document:", response_of_document)
        #             # if main_path
        #
        #             # 完善器-知识图谱
        #             graph_optimization_result = vertex_evaluator.process_vertices(sub_question, None,
        #                                                                           main_path=single_path,
        #                                                                           nei_path=neighbor_list)
        #             # print(graph_optimization_result)
        #             # 文档检索
        #             with open('hotpotqa/questions_and_answers.json', "r", encoding='utf-8') as qas:
        #                 # 将 JSON 字符串解析为 Python 对象
        #                 q_as = json.loads(qas.read())
        #                 text = q_as[i - 1]['context']
        #                 documents = document_search(root_path, sub_question, text, graph_optimization_result)
        #             # documents = document_search(root_path,question)
        #             #     print("文档检索结果:", documents)
        #             # 完善器-文档
        #             # document_optimization_result = vertex_evaluator.process_vertices(sub_question,Steps, text_vertices=documents)
        #             # print("完善器-文档:",document_optimization_result)
        #
        #             # 将每个子问题推理出答案
        #             # # 8. answer generation 子问题答案生成
        #             sub_answer = generated_sub_answer(sub_qa_list, sub_question, graph_optimization_result,
        #                                               documents)
        #             # print("     sub_answer:",sub_answer)
        #             combined_dict = {
        #                 sq_num: sub_question,
        #                 'sub_answer': sub_answer
        #             }
        #             sub_qa_list.append(combined_dict)
        #             # json.dump(combined_dict, sbqa, indent=4)
        #         # all_sub.append(sub_qa)
        #         qa_list.append(sub_qa_list)
        #         yn = ("yes" in answer or "no" in answer)
        #         fianlanswer = all_answers(yn,sub_question, sub_qa_list)
        #         # print('1-fianlanswer:', fianlanswer)
        #         # if fianlanswer.lower() in answer.lower() or answer.lower() in fianlanswer.lower():
        #         #     fianlanswer = answer_refine(question, fianlanswer)
        #         # print('2-fianlanswer:', fianlanswer)
        #         # if fianlanswer == 'Unknown' or fianlanswer.lower() in answer.lower() or answer.lower() in fianlanswer.lower():
        #         #         fianlanswer = answer_retrieve(question, text)
        #         print('fianlanswer:', fianlanswer)
        #         print("========================================")
        #         # 将每个问题的答案和HGot-RAG答案存储
        #         # oneqa_dict = {
        #         #     sq_num: question,
        #         #     'answer': all_sub,
        #         #     'ground_truth': answer
        #         # }
        #         # qa_dict ={
        #         #     'question': question,
        #         #     'answer': fianlanswer,
        #         #     'ground_truth': answer
        #         # }
        #         qa_dict = {
        #             'question': question,
        #             'answer': fianlanswer,
        #             'ground_truth': answer
        #         }
        #         # print('\nHGot-RAG:\n', fianlanswer)
        #         # # 构建新的JSON数据
        #         result.append(qa_dict)
        #         # json.dump(qa_dict,sbqa, indent=4)
        #
        # # 将新的JSON数据写入文件
        # with open('hotpotqa/reslt.json', 'w') as file:
        #     json.dump(result, file, indent=4)
        # 9. Experiment 1: chatgpt
        # try:
    # # 将新的JSON数据写入文件
    # with open('hotpotqa/reslt_ollama.json', 'w') as file:
    #     with open('hotpotqa/reslt_document.json', 'w') as file:

            # chatgpt_result = chat_35(str(input_text[0]))
                chatgpt_result = chat_ollama(question)
                print('\nchat_ollama:',chatgpt_result)
                ollama_dict = {
                        'question': question,
                        'answer': chatgpt_result,
                        'ground_truth': answer
                    }
                ollama_qa.append(ollama_dict)


        # document

                # document_result = document_search(root_path, question, text=None, refined_kg=None)
                # print('\ndocument_qa:', document_qa)
                # document_dict = {
                #     'question': question,
                #     'answer': document_result,
                #     'ground_truth': answer
                # }
                # document_qa.append(document_dict)
    with open('2wiki/reslt_ollama.json', 'w') as file:
        json.dump(ollama_qa, file, indent=4)
    # with open('2wiki/reslt_document.json', 'w') as file:
    #         # # 将新的JSON数据写入文件
    #         json.dump(document_qa, file, indent=4)

