import json
import ast
import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from openai import OpenAI
import faiss
import time
import re
from sklearn.metrics.pairwise import cosine_similarity
# from openai.types.shared_params import response_format_json_object
from openai.types.chat.completion_create_params import ResponseFormatJSONObject


# 三元组版本v1.0
# template="""
#     Instruction: "Please deconstruct the main question into a sequence of precise sub-questions that will systematically lead to the collection of necessary information. Each sub-question should be designed to extract a specific piece of information necessary for answering the main question without any superfluous details. The sub-questions should culminate in a set of collected information that can be used to answer the main question.\n\n        Detail the steps required to address each sub-question and identify any dependencies among them. If a sub-question is contingent upon the answer of another, clearly articulate how they are interconnected. Ensure that the sub-questions can be answered using the information available from the search described in the main question, and that no additional information is sought from the user.\n        Make sure the words different problems are consistent. Strictly adhere to the following template for your response, and include no extraneous output.don not Note"
#     <Examples>
#
#         Input: "Do both films: The Ex-Mrs. Bradford and The Star Of Santa Clara have the directors from the same country?"
#         Output: [
#     {
#         "Sub-question 1": "(The Ex-Mrs. Bradford, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Ex-Mrs. Bradford'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 2": "(The Star Of Santa Clara, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Star Of Santa Clara'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 3": "(Sub-question 1, COUNTRY_OF_ORIGIN, ?)",
#         "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 1.",
#         "Dependencies": "Sub-question 1"
#     },
#     {
#         "Sub-question 4": "(Sub-question 2, COUNTRY_OF_ORIGIN, ?)",
#         "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 2.",
#         "Dependencies": "Sub-question 2"
#     }
# ]
# Input: "Which film has the director who was born later, The Great Dictator or The Philadelphia Story?"
# Output: [
#     {
#         "Sub-question 1": "(The Great Dictator, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Great Dictator'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 2": "(The Philadelphia Story, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Philadelphia Story'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 3": "(Sub-question 1, DATE_OF_BIRTH, ?)",
#         "Steps to Answer": "Search for the birth date of the director identified in Sub-question 1.",
#         "Dependencies": "Sub-question 1"
#     },
#     {
#         "Sub-question 4": "(Sub-question 2, DATE_OF_BIRTH, ?)",
#         "Steps to Answer": "Search for the birth date of the director identified in Sub-question 2.",
#         "Dependencies": "Sub-question 2"
#     }
# ]
#     </Examples>
#      Relationships: [
#         "AWARD_RECEIVED",
#         "CAUSE_OF_DEATH",
#         "CHILD",
#         "COMPOSER",
#         "COUNTRY",
#         "COUNTRY_OF_CITIZENSHIP",
#         "COUNTRY_OF_ORIGIN",
#         "CREATOR",
#         "DATE_OF_BIRTH",
#         "DATE_OF_DEATH",
#         "DIRECTOR",
#         "DOCTORAL_ADVISOR",
#         "EDITOR",
#         "EDUCATED_AT",
#         "EMPLOYER",
#         "FATHER",
#         "FOUNDED_BY",
#         "HAS_PART",
#         "INCEPTION",
#         "MANUFACTURER",
#         "MOTHER",
#         "OCCUPATION",
#         "PERFORMER",
#         "PLACE_OF_BIRTH",
#         "PLACE_OF_BURIAL",
#         "PLACE_OF_DEATH",
#         "PLACE_OF_DETENTION",
#         "PRESENTER",
#         "PRODUCER",
#         "PUBLICATION_DATE",
#         "PUBLISHER",
#         "SIBLING",
#         "SPOUSE",
#         "STUDENT_OF"
#     ]
# 三元组版本V2.0
#     template = """Please deconstruct the main question into a sequence of precise sub-questions that will systematically lead to the collection of necessary information. Each sub-question should be designed to extract a specific piece of information necessary for answering the main question without any superfluous details. The sub-questions should culminate in a set of collected information that can be used to answer the main question.Return json format.
#
# 1. Sub-question Design: Each sub-question should aim to extract a specific piece of information required to answer the main question.
# 2. Information Collection: Ensure that the sub-questions can be answered using the information available from the search described in the main question, without seeking additional information from the user.
# 3. Steps and Dependencies: Detail the steps required to address each sub-question and identify any dependencies among them. If a sub-question is contingent upon the answer of another, clearly articulate how they are interconnected.
# 4. Consistency: Ensure that the terms 'different problems' are used consistently.
# 5. Template Adherence: Strictly adhere to the following template for your response, and avoid including any extraneous output.
#
# Examples:
#
# Input: 'Do both films: The Ex-Mrs. Bradford and The Star Of Santa Clara have directors from the same country?'
# Output: [
#     {
#         "Sub-question 1": "(The Ex-Mrs. Bradford, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Ex-Mrs. Bradford'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 2": "(The Star Of Santa Clara, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Star Of Santa Clara'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 3": "(Sub-question 1, COUNTRY_OF_ORIGIN, ?)",
#         "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 1.",
#         "Dependencies": "Sub-question 1"
#     },
#     {
#         "Sub-question 4": "(Sub-question 2, COUNTRY_OF_ORIGIN, ?)",
#         "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 2.",
#         "Dependencies": "Sub-question 2"
#     }
# ]
#
# Input: 'Which film has the director who was born later, The Great Dictator or The Philadelphia Story?'
# Output: [
#     {
#         "Sub-question 1": "(The Great Dictator, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Great Dictator'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 2": "(The Philadelphia Story, DIRECTOR, ?)",
#         "Steps to Answer": "Search for the director of the film 'The Philadelphia Story'.",
#         "Dependencies": "None"
#     },
#     {
#         "Sub-question 3": "(Sub-question 1, DATE_OF_BIRTH, ?)",
#         "Steps to Answer": "Search for the birth date of the director identified in Sub-question 1.",
#         "Dependencies": "Sub-question 1"
#     },
#     {
#         "Sub-question 4": "(Sub-question 2, DATE_OF_BIRTH, ?)",
#         "Steps to Answer": "Search for the birth date of the director identified in Sub-question 2.",
#         "Dependencies": "Sub-question 2"
#     }
# ]
#
# Select the relation from the list below instead of generating and return empty if there is no suitable choice, such as using COUNTRY_OF_ORIGIN from the list instead of LOCATION: [AWARD_RECEIVED, CAUSE_OF_DEATH, CHILD, COMPOSER, COUNTRY, COUNTRY_OF_CITIZENSHIP, COUNTRY_OF_ORIGIN, CREATOR, DATE_OF_BIRTH, DATE_OF_DEATH, DIRECTOR, DOCTORAL_ADVISOR, EDITOR, EDUCATED_AT, EMPLOYER, FATHER, FOUNDED_BY, HAS_PART, INCEPTION, MANUFACTURER, MOTHER, OCCUPATION, PERFORMER, PLACE_OF_BIRTH, PLACE_OF_BURIAL, PLACE_OF_DEATH, PLACE_OF_DETENTION, PRESENTER, PRODUCER, PUBLICATION_DATE, PUBLISHER, SIBLING, SPOUSE, STUDENT_OF]


# 实体匹配
def call_embedding_api(entities, max_retries=3):
    embedding_dict = {}
    failed_entities = entities

    while failed_entities and max_retries > 0:
        new_failed_entities = set()

        for entity in failed_entities:
            try:
                response = ollama.embed(model="nomic-embed-text", input=entity)
                embedding = response["embeddings"][0]
                embedding_dict[entity] = embedding
            except Exception as e:
                print(f"<Warning> no embedding found for {entity} or error occurred: {e}")
                new_failed_entities.add(entity)

        failed_entities = new_failed_entities
        max_retries -= 1

    if failed_entities:
        print(f"<Warning> The following entities still failed after {max_retries} retries: {failed_entities}")

    return embedding_dict


def load_entities(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["entity", "embedding"])
    return df


def save_embeddings_batch(embeddings, file_path, batch_size=1000):
    if os.path.exists(file_path):
        existing_embeddings = load_embeddings(file_path)
        embeddings.update(existing_embeddings)

    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = dict(list(embeddings.items())[start:end])

        with open(file_path, 'a') as f:
            for entity, embedding in batch_embeddings.items():
                f.write(json.dumps({entity: embedding}) + '\n')


def load_embeddings(file_path):
    embeddings = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                embeddings.update(entry)
    return embeddings


# def process_question_kg(question_embeddings, entity_embeddings):
#     """
#     处理问题和知识图谱的实体匹配。
#
#     通过计算问题嵌入和实体嵌入之间的余弦相似性，找到最匹配的实体，并将其添加到匹配结果中。
#     如果在添加实体时发生冲突（即实体已在匹配结果中），则选择次优的匹配实体。
#
#     参数:
#     - question_embeddings: 问题的嵌入表示，字典类型，键为问题实体，值为嵌入向量。
#     - entity_embeddings: 知识图谱实体的嵌入表示，字典类型，键为实体名称，值为嵌入向量。
#
#     返回:
#     - match_kg: 匹配到的实体列表。
#     """
#     match_kg = []  # 用于存储匹配到的实体
#
#     # 将entity_embeddings转换为DataFrame，便于后续操作
#     entity_embeddings_df = pd.DataFrame(list(entity_embeddings.items()), columns=['entity', 'embedding'])
#     # 将嵌入向量转换为numpy数组
#     entity_embeddings_df['embedding'] = entity_embeddings_df['embedding'].apply(lambda x: np.array(x))
#
#     # 遍历问题的嵌入，为每个问题嵌入找到最匹配的实体
#     for kg_entity, kg_embedding in question_embeddings.items():
#         try:
#             kg_embedding = np.array(kg_embedding)  # 将问题嵌入转换为numpy数组
#
#             # 初始化最大余弦相似性和对应的实体索引
#             max_cos_similarity = -1
#             max_index = None
#
#             # 遍历每个实体嵌入，计算余弦相似性
#             for index, entity_embedding in enumerate(entity_embeddings_df['embedding']):
#                 cos_similarity = cosine_similarity(entity_embedding.reshape(1, -1), kg_embedding.reshape(1, -1))[0][0]
#                 if cos_similarity > max_cos_similarity:
#                     max_cos_similarity = cos_similarity
#                     max_index = index
#
#             # 获取最匹配的实体
#             match_kg_i = entity_embeddings_df['entity'][max_index]
#
#             # 如果匹配到的实体已经在结果列表中，则寻找次优实体
#             while match_kg_i in match_kg:
#                 # 将当前最大索引的相似性置为0
#                 entity_embeddings_df.at[max_index, 'embedding'] = np.zeros_like(entity_embeddings_df['embedding'][max_index])
#                 # 重新计算最大相似性
#                 max_cos_similarity = -1
#                 max_index = None
#                 for index, entity_embedding in enumerate(entity_embeddings_df['embedding']):
#                     cos_similarity = cosine_similarity(entity_embedding.reshape(1, -1), kg_embedding.reshape(1, -1))[0][0]
#                     if cos_similarity > max_cos_similarity:
#                         max_cos_similarity = cos_similarity
#                         max_index = index
#                 match_kg_i = entity_embeddings_df['entity'][max_index]
#
#             # 将匹配到的实体添加到结果列表中
#             match_kg.append(match_kg_i)
#         except Exception as e:
#             print(f"<Warning> no entities found or error occurred: {e}")  # 打印警告信息
#             continue  # 继续处理下一个问题嵌入
#
#     return match_kg
def process_question_kg(question_embeddings, entity_embeddings):
    """
    处理问题和知识图谱的实体匹配。

    通过计算问题嵌入和实体嵌入之间的余弦相似性，找到最匹配的实体，并将其添加到匹配结果中。
    如果在添加实体时发生冲突（即实体已在匹配结果中），则选择次优的匹配实体。

    参数:
    - question_embeddings: 问题的嵌入表示，字典类型，键为问题实体，值为嵌入向量。
    - entity_embeddings: 知识图谱实体的嵌入表示，字典类型，键为实体名称，值为嵌入向量。

    返回:
    - match_kg: 匹配到的实体列表。
    """
    start_time = time.time()  # 开始计时
    match_kg = []  # 用于存储匹配到的实体

    # 将entity_embeddings转换为DataFrame，便于后续操作
    entity_embeddings_df = pd.DataFrame(list(entity_embeddings.items()), columns=['entity', 'embedding'])
    # 将嵌入向量转换为numpy数组
    entity_embeddings_df['embedding'] = entity_embeddings_df['embedding'].apply(lambda x: np.array(x))

    # 将所有实体嵌入转换为numpy数组
    embeddings_array = np.vstack(entity_embeddings_df['embedding'].tolist())

    # 创建Faiss索引
    d = embeddings_array.shape[1]  # 嵌入维度
    index = faiss.IndexFlatIP(d)  # 创建索引，IP表示内积
    index.add(embeddings_array)  # 添加所有实体嵌入到索引中

    # 遍历问题的嵌入，为每个问题嵌入找到最匹配的实体
    for kg_entity, kg_embedding in question_embeddings.items():
        try:
            kg_embedding = np.array(kg_embedding).reshape(1, -1)  # 将问题嵌入转换为numpy数组

            # 查找最相似的实体
            D, I = index.search(kg_embedding, k=1)  # 搜索最相似的前10个实体
            matched_indices = I.flatten()  # 获取索引
            matched_similarities = D.flatten()  # 获取相似性得分

            # 选择最匹配的实体
            max_cos_similarity = -1
            max_index = None
            for idx, sim in zip(matched_indices, matched_similarities):
                if sim > max_cos_similarity:
                    max_cos_similarity = sim
                    max_index = idx

            match_kg_i = entity_embeddings_df['entity'][max_index]

            # 如果匹配到的实体已经在结果列表中，则寻找次优实体
            while match_kg_i in match_kg:
                # 将当前最大索引的相似性置为0
                embeddings_array[max_index] = np.zeros_like(embeddings_array[max_index])
                index.add(embeddings_array)  # 更新索引

                # 重新计算最大相似性
                D, I = index.search(kg_embedding, k=10)
                matched_indices = I.flatten()
                matched_similarities = D.flatten()

                max_cos_similarity = -1
                max_index = None
                for idx, sim in zip(matched_indices, matched_similarities):
                    if sim > max_cos_similarity:
                        max_cos_similarity = sim
                        max_index = idx

                match_kg_i = entity_embeddings_df['entity'][max_index]

            # 将匹配到的实体添加到结果列表中
            match_kg.append(match_kg_i)
        except Exception as e:
            print(f"<Warning> no entities found or error occurred: {e}")  # 打印警告信息
            continue  # 继续处理下一个问题嵌入
    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    return match_kg


def match_entities(entities_file_path, embeddings_file_path, query_entities):
    """
    匹配查询实体与知识图谱中的实体。

    首先加载实体和对应的嵌入表示，如果嵌入文件不存在则调用嵌入API生成。
    然后调用process_question_kg函数处理问题嵌入与实体嵌入的匹配。

    参数:
    - entities_file_path: 实体文件路径。
    - embeddings_file_path: 嵌入文件路径。
    - query_entities: 查询实体列表。

    返回:
    - 匹配到的实体列表。
    """
    # 加载知识图谱实体
    entities_df = load_entities(entities_file_path)
    entities = entities_df['entity'].tolist()  # 将实体转换为列表

    # 加载实体嵌入表示
    embeddings = load_embeddings(embeddings_file_path)
    # 如果没有加载到嵌入，则调用嵌入API生成并保存
    if not embeddings:
        print("No embeddings found, generating...")
        embeddings = call_embedding_api(entities)
        save_embeddings_batch(embeddings, embeddings_file_path)

    # 调用嵌入API生成查询实体的嵌入表示
    question_embeddings = call_embedding_api(query_entities)

    # 处理问题嵌入与实体嵌入的匹配，获取匹配到的实体列表
    entities = process_question_kg(question_embeddings, embeddings)
    return entities


# Input: "Do both films: The Ex-Mrs. Bradford and The Star Of Santa Clara have the directors from the same country?"
#
# Output:
# [
#     {
#     "Sub-question 1": "Who is the director of The Ex-Mrs. Bradford?",
#     "Steps to Answer": "Search for the director of the film 'The Ex-Mrs. Bradford'.",
#     "Purpose": "To identify the director of the first film for comparison.",
#     "Dependencies": "None"
#     },
#     {
#     "Sub-question 2": "Who is the director of The Star Of Santa Clara?",
#     "Steps to Answer": "Search for the director of the film 'The Star Of Santa Clara'.",
#     "Purpose": "To identify the director of the second film for comparison.",
#     "Dependencies": "None"
#     },
#     {
#     "Sub-question 3": "What is the country of origin of the director of The Ex-Mrs. Bradford?",
#     "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 1.",
#     "Purpose": "To determine the country of origin of the director of the first film for comparison.",
#     "Dependencies": "Sub-question 1"
#     },
#     {
#     "Sub-question 4": "What is the country of origin of the director of The Star Of Santa Clara?",
#     "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 2.",
#     "Purpose": "To determine the country of origin of the director of the second film for comparison.",
#     "Dependencies": "Sub-question 2"
#     },
#     {
#     "Sub-question 5": "Are the countries of origin of the two directors the same?",
#     "Steps to Answer": "Compare the countries of origin obtained from Sub-questions 3 and 4 to determine if they are identical.",
#     "Purpose": "To answer the original question by confirming if both films have directors from the same country.",
#     "Dependencies": "Sub-question 3, Sub-question 4"
#     }
# ]
# "Please analyze the question provided below and create a concise set of sub-questions that directly address key aspects of the main inquiry. Focus on identifying the core components of the question without adding any new elements.
# Ensure that each sub-question is essential for answering the overarching question and avoid unnecessary elaboration. Subquestions can be answered with the search information described in the question, but the patient cannot be asked for more information.
# List the steps you need to take to solve each subproblem, and specify the dependencies between the subproblems. If there are dependencies, explain them in detail.The following template is strictly followed, and nothing else is output."
# 子问题生成
def prompt_to_subquestions(input_text):
    template = """
        <Instruction>
        "Please deconstruct the main question into a sequence of precise sub-questions that will systematically lead to the resolution of the main inquiry. Each sub-question should be designed to extract a specific piece of information necessary for answering the main question without any superfluous details. The sequence should culminate in a final sub-question whose answer directly provides the solution to the main question.

        Detail the steps required to address each sub-question and identify any dependencies among them. If a sub-question is contingent upon the answer of another, clearly articulate how they are interconnected. Ensure that the sub-questions can be answered using the information available from the search described in the main question, and that no additional information is sought from the user.
        Make sure the words different problems are consistent.Strictly adhere to the following template for your response, and include no extraneous output."

        Template:
        [
        {
        "Sub-question": "A concise sub-question that targets a distinct component of the main inquiry.",
        "Steps to Answer": "A clear, step-by-step process on how to find the answer to the sub-question.",
        "Purpose": "An explanation of how this sub-question brings us closer to answering the main question, with the final sub-question's answer directly answering the main question.",
        "Dependencies": "A list of preceding sub-questions that need to be resolved before this one can be addressed, if applicable."
        }
        ]
        </Instruction>
        <Examples>
        Input: "Do both films: The Ex-Mrs. Bradford and The Star Of Santa Clara have the directors from the same country?"
        Output:
        [
            {
            "Sub-question 1": "Who is the director of The Ex-Mrs. Bradford?",
            "Steps to Answer": "Search for the director of the film 'The Ex-Mrs. Bradford'.",
            "Purpose": "To identify the director of the first film for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 2": "Who is the director of The Star Of Santa Clara?",
            "Steps to Answer": "Search for the director of the film 'The Star Of Santa Clara'.",
            "Purpose": "To identify the director of the second film for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 3": "What is the country of the director of The Ex-Mrs. Bradford?",
            "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 1.",
            "Purpose": "To determine the country of origin of the director of the first film for comparison.",
            "Dependencies": "Sub-question 1"
            },
            {
            "Sub-question 4": "What is the country of the director of The Star Of Santa Clara?",
            "Steps to Answer": "Search for the country of origin of the director identified in Sub-question 2.",
            "Purpose": "To determine the country of origin of the director of the second film for comparison.",
            "Dependencies": "Sub-question 2"
            },
            {
            "Sub-question 5": "Are the countries of the two directors the same?",
            "Steps to Answer": "Compare the countries of origin obtained from Sub-questions 3 and 4 to determine if they are identical.",
            "Purpose": "To answer the original question by confirming if both films have directors from the same country.",
            "Dependencies": "Sub-question 3, Sub-question 4"
            }
        ]
        Input: "Which film has the director who was born later, The Great Dictator or The Philadelphia Story?"
        Output:
        [
            {
            "Sub-question 1": "Who is the director of The Great Dictator?",
            "Steps to Answer": "Search for the director of the film 'The Great Dictator'.",
            "Purpose": "To identify the director of the first film for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 2": "Who is the director of The Philadelphia Story?",
            "Steps to Answer": "Search for the director of the film 'The Philadelphia Story'.",
            "Purpose": "To identify the director of the second film for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 3": "When was the director of The Great Dictator born?",
            "Steps to Answer": "Search for the birth date of the director identified in Sub-question 1.",
            "Purpose": "To determine the birth year of the director of the first film for comparison.",
            "Dependencies": "Sub-question 1"
            },
            {
            "Sub-question 4": "When was the director of The Philadelphia Story born?",
            "Steps to Answer": "Search for the birth date of the director identified in Sub-question 2.",
            "Purpose": "To determine the birth year of the director of the second film for comparison.",
            "Dependencies": "Sub-question 2"
            },
            {
            "Sub-question 5": "Compare the birth years of the two directors. Which director was born later?",
            "Steps to Answer": "Compare the birth years obtained from Sub-questions 3 and 4 to determine which director was born later.",
            "Purpose": "To answer the original question by identifying which film has the director who was born later.",
            "Dependencies": "Sub-question 3, Sub-question 4"
            },
            {
            "Sub-question 6": "What is the title of the film directed by the director born later?",
            "Steps to Answer": "Match the film title to the director identified in Sub-question 5.",
            "Purpose": "To provide the title of the film that has the director who was born later.",
            "Dependencies": "Sub-question 5"
            }
        ]


        Input: "Do the Amazon River and the Nile River both flow into the same ocean?"
        Output:
        [
            {
            "Sub-question 1": "Which ocean does the Amazon River flow into?",
            "Steps to Answer": "Search for the ocean into which the Amazon River ultimately drains.",
            "Purpose": "To identify the ocean associated with the first river for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 2": "Which ocean does the Nile River flow into?",
            "Steps to Answer": "Search for the ocean into which the Nile River ultimately drains.",
            "Purpose": "To identify the ocean associated with the second river for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 3": "Are the oceans identified in Sub-questions 1 and 2 the same?",
            "Steps to Answer": "Compare the oceans obtained from Sub-questions 1 and 2 to determine if they are the same.",
            "Purpose": "To answer the original question by confirming if both rivers flow into the same ocean.",
            "Dependencies": "Sub-question 1, Sub-question 2"
            }
        ]
        Input: "Which scientist has more publications, Albert Einstein or Isaac Newton?"
        Output:
        [
            {
            "Sub-question 1": "What is the total number of publications by Albert Einstein?",
            "Steps to Answer": "Search for the total count of scientific publications authored by Albert Einstein.",
            "Purpose": "To determine the publication count of the first scientist for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 2": "What is the total number of publications by Isaac Newton?",
            "Steps to Answer": "Search for the total count of scientific publications authored by Isaac Newton.",
            "Purpose": "To determine the publication count of the second scientist for comparison.",
            "Dependencies": "None"
            },
            {
            "Sub-question 3": "Compare the publication counts of Albert Einstein and Isaac Newton.",
            "Steps to Answer": "Compare the publication counts obtained from Sub-questions 1 and 2 to determine who has more publications.",
            "Purpose": "To answer the original question by identifying which scientist has more publications.",
            "Dependencies": "Sub-question 1, Sub-question 2"
            }
        ]

    </Examples>
    Input: """ + input_text + """
    Output:
    """

    # 构建消息列表
    messages = [
        {"role": "system", "content": template}
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        seed=123,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=0,
        n=1)
    response_of_subjects = res.choices[0].message.content

    response_of_subjects = clean_json_string(response_of_subjects)
    print(f"Response of subjects: {response_of_subjects}")
    try:
        #     return json.loads(response_of_subjects)
        return json.loads(response_of_subjects)
    except json.JSONDecodeError as e:
        # print(f"Error parsing JSON: {e}")
        # 尝试修复 JSON 字符串
        fixed_json_str = fix_json_format(response_of_subjects)
        try:
            data = json.loads(fixed_json_str)
            print("JSON has been fixed.")
            return data
        except json.JSONDecodeError as e:
            print(f"Failed to fix JSON: {e}")
            return None
        return json.loads(response_of_subjects)



def fix_json_format(json_str):
    # 使用正则表达式将单引号替换为双引号
    fixed_json_str = re.sub(r"'(?!')", '"', json_str)

    # 确保所有属性名都使用双引号
    fixed_json_str = re.sub(r'(\w+):', r'"\1":', fixed_json_str)

    return fixed_json_str

def clean_json_string(json_str):
    # 清理非法的转义字符
    cleaned_json_str = re.sub(r"'", '', json_str)
    # sub=re.findall(r"\[(.*?)\]", cleaned_json_str)
    # print(sub)
    # cleaned_json_str = "["+sub+"]"
    cleaned_json_str="["+cleaned_json_str.split("[")[1].split("]")[0]+"]"
    # print(cleaned_json_str)
    return cleaned_json_str


def extract_entities(input_text):
    template = """<Instruction> Identify all relevant entities from the following question. Your output should consist solely of entity names, with each entity on a separate line, and no explanations or descriptions should be included.

            Input: [input]
            output:
            ['Entity1', Entity2', Entity3','...']
            </Instruction>

            <Examples>
            Example 0
            Question: What are the impacts of global warming on the habitat of polar bears?
            Output:['Global Warming',Polar Bears','Habitat,Impacts']

            Example 1
            Input:What were the main causes of World War II?
            Output:['World War II', 'Main Causes']

            Example 2
            Input:Who directed the movie Inception and what are its main themes?
            Output:['Inception', 'Directed', 'Main Themes']

            Example 3
            Input:How does climate change affect agricultural productivity in Sub-Saharan Africa?
            Output:['Climate Change', 'Agricultural Productivity', 'Sub-Saharan Africa']
            </Examples>

            Input:
            {input}
            Output:
            """.format(input=input_text)
    messages = [
        {"role": "system", "content": template},
    ]
    res = client.chat.completions.create(
        model=ollama_model,
        seed=123,
        messages=messages,
        temperature=0,
        stream=False,
        top_p=1,
        n=1,
        # response_format={"type": "json_object"} # 设置响应格式为 JSON 对象
    )
    # print(type(res))
    response_of_subjects = res.choices[0].message.content
    # print("response_of_subjects:",response_of_subjects)
    return convert_str_to_list(response_of_subjects)


# def convert_str_to_list(input_str):
#     # 使用 ast.literal_eval 安全地将字符串转换为列表
#     result_list = ast.literal_eval(input_str)
#     return result_list
def convert_str_to_list(input_str):
    # print("input_str:",input_str)
    try:
        # 尝试直接解析
        result_list = ast.literal_eval(input_str)
        return result_list
    except (ValueError, SyntaxError):
        # 如果直接解析失败，尝试进行预处理
        # cleaned_input_str = input_str.strip().replace("'", '"')
        # 去除非法转义字符
        # cleaned_response = re.sub(r'\\u[0-9a-fA-F]{0,4}', '', input_str)
        try:
            # 再次尝试解析
            result_list = json.loads(input_str)
            return result_list
        except json.JSONDecodeError:
            # 如果仍然解析失败，返回空列表
            print(f"Failed to parse input string: {input_str}")
            return input_str


def main():
    # 读取QA JSON文件
    # with open('2wiki/questions_and_answers.json', 'r', encoding='utf-8') as file:
    # with open('hotpotqa/bc.json', 'r', encoding='utf-8') as file:
    with open('hotpotqa/bc.json', 'r') as file:
        qa_data = json.load(file)

    # 处理每个问题
    new_qa_data = []
    i = 0
    for item in qa_data:
        i = i + 1
        # if i >= 10:
        #     break
        print("第", i, "个问题")
        question = item['question'].replace("'", "`")
        answer = item['ground_truth']

        # 实体提取
        # entities = extract_entities(question)  # 你需要实现这个函数
        # print("实体提取",entities)
        # 相似度匹配
        # matched_entities = match_entities("download/raw_data/2wikimultihopqa/entity2id.txt", "download/raw_data/2wikimultihopqa/embeddings.json", entities)
        # print("相似度匹配", matched_entities)
        # 生成子问题
        sub_questions = prompt_to_subquestions(question)
        # print("sub_questions:", sub_questions)
        # print("生成子问题", sub_questions)
        new_sub_questions = []
        for sq in sub_questions:
            print("sq:", sq)
            # 初始化空列表
            try:
                sub_question = sq[next(key for key in sq if key.startswith("Sub-question"))]
                # print("question:", sub_question)
            except StopIteration:
                print("第", i, "个问题未找到以 'Sub-question' 开头的键,生成失败")
                break
            except Exception as e:
                print(f"发生其他错误: {e}")
                break

            # head_entity = sub_question.split(",")[0].split("(")[1]
            # if (head_entity.startswith("Sub-question")):
            #     break
            # relation = sub_question.split(",")[1]
            # print("head_entity:", head_entity)
            # print("relation:", relation)
            # question_kg = Entity_recognition(sub_question)
            question_kg = extract_entities(sub_question)
            # print("question_kg:", question_kg)
            # 实体对齐
            # head_entity = match_entities("2wiki/entity2id.txt", '2wiki/embeddings.json',head_entity)
            match_kg = question_kg
            sq["match_kg"] = match_kg
            # sq["match_kg"] = head_entity, relation
            new_sub_questions.append(sq)

        # 构建新的JSON数据
        new_qa_data.append({
            "question": question,
            "answer": answer,
            # "entities": matched_entities,
            # "entities": entities,
            "sub_questions": new_sub_questions
        })

    # 将新的JSON数据写入文件
    # with open('F:/code/HToG-RAG/2wiki/qaes.json', 'w') as file:
    with open('F:/code/HToG-RAG/hotpotqa/补充_qaes.json', 'w') as file:
        json.dump(new_qa_data, file, indent=4)


OPENAI_API_KEY = "empty"
base_url = "http://127.0.0.1:11434/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)
ollama_model = 'qwen2.5:7b-instruct'
# ollama_model = 'llama3-chatqa'
if __name__ == "__main__":
    main()
