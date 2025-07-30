import ast
import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

def call_embedding_api(entities, max_retries=3):
    """
    调用API获取实体的嵌入向量。
    :param entities: 实体列表
    :param max_retries: 最大重试次数
    :return: 包含实体及其嵌入向量的字典
    """
    embedding_dict = {}
    failed_entities = set(entities)  # 初始失败实体集合

    while failed_entities and max_retries > 0:
        new_failed_entities = set()  # 当前重试失败的实体集合

        for entity in failed_entities:
            try:
                # API接口可以获取嵌入向量
                # API调用
                response = ollama.embed(model="nomic-embed-text", input=entity)
                embedding = response["embeddings"][0]
                embedding_dict[entity] = embedding
            except Exception as e:
                print(f"<Warning> no embedding found for {entity} or error occurred: {e}")
                new_failed_entities.add(entity)  # 将失败的实体添加到新的失败实体集合中

        failed_entities = new_failed_entities  # 更新失败实体集合
        max_retries -= 1  # 减少重试次数

    if failed_entities:
        print(f"<Warning> The following entities still failed after {max_retries} retries: {failed_entities}")

    return embedding_dict

def load_entities(file_path):
    """
    从文件中加载实体及其ID。
    :param file_path: 文件路径
    :return: 包含实体和ID的DataFrame
    """
    df = pd.read_csv(file_path, sep="\t", header=None, names=["entity", "id"])
    return df

def save_embeddings_batch(embeddings, file_path='embeddings.json', batch_size=1000):
    """
    分批保存嵌入向量到文件。
    :param embeddings: 包含实体及其嵌入向量的字典
    :param file_path: 保存文件的路径
    :param batch_size: 每次保存的批量大小
    """
    if os.path.exists(file_path):
        existing_embeddings = load_embeddings(file_path)
        embeddings.update(existing_embeddings)

    num_batches = (len(embeddings) + batch_size - 1) // batch_size  # 向上取整
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(embeddings))
        batch_embeddings = dict(list(embeddings.items())[start:end])

        with open(file_path, 'a') as f:
            for entity, embedding in batch_embeddings.items():
                f.write(json.dumps({entity: embedding}) + '\n')

# def load_embeddings(file_path='embeddings.json'):
    """
    从文件加载嵌入向量。
    :param file_path: 加载文件的路径
    :return: 包含实体及其嵌入向量的字典
    """
    # embeddings = {}
    # if os.path.exists(file_path):
    #     with open(file_path, 'r') as f:
    #         for line in f:
    #             entry = json.loads(line.strip())
    #             embeddings.update(entry)
    # return embeddings
def load_embeddings(embeddings_file_path):
    with open(embeddings_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 移除文件末尾多余的字符
    if lines[-1].strip() == '':
        lines.pop()

    embeddings = []
    for line in lines:
        try:
            entry = json.loads(line.strip())
            embeddings.append(entry)
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line}")
            continue

    return embeddings

def process_question_kg(question_embeddings, entity_embeddings):
    """
    对给定的question_kg进行处理，找到最相似的实体。
    :param question_embeddings: 包含问题实体及其嵌入向量的字典
    :param entity_embeddings: 包含实体及其嵌入向量的字典
    :return: 匹配的实体列表
    """
    match_kg = []

    # 将实体嵌入向量转换为DataFrame
    entity_embeddings_list = []
    for item in entity_embeddings:
        for entity, embedding in item.items():
            entity_embeddings_list.append({'entity': entity, 'embedding': embedding})

    entity_embeddings_df = pd.DataFrame(entity_embeddings_list)
    entity_embeddings_df['embedding'] = entity_embeddings_df['embedding'].apply(lambda x: np.array(x))

    # 确保所有的嵌入向量都是一维数组
    assert all(embedding.ndim == 1 for embedding in entity_embeddings_df['embedding']), "Embeddings must be 1D arrays"

    for kg_entity, kg_embedding in question_embeddings.items():
        kg_embedding = np.array(kg_embedding)

        # 计算余弦相似度
        cos_similarities = cosine_similarity(entity_embeddings_df['embedding'].tolist(), kg_embedding.reshape(1, -1)).flatten()
        max_index = cos_similarities.argmax()

        match_kg_i = entity_embeddings_df['entity'][max_index]
        while match_kg_i.replace(" ", "_") in match_kg:
            cos_similarities[max_index] = 0
            max_index = cos_similarities.argmax()
            match_kg_i = entity_embeddings_df['entity'][max_index]

        match_kg.append(match_kg_i.replace(" ", "_"))

    return match_kg

def match_entities(entities_file_path, embeddings_file_path, query_entities):
    """
    匹配问题KG中的实体。
    :param entities_file_path: 实体文件路径
    :param embeddings_file_path: 嵌入向量文件路径
    :param query_entities: 查询实体列表
    :return: 匹配的实体列表
    """
    # 从实体文件中加载实体
    entities_df = load_entities(entities_file_path)
    entities = entities_df['entity'].tolist()
    # 检查是否已有嵌入向量文件
    embeddings = load_embeddings(embeddings_file_path)
    if not embeddings:
        # 如果没有，则获取所有实体的嵌入向量并保存
        embeddings = call_embedding_api(entities)
        save_embeddings_batch(embeddings, embeddings_file_path)

    # 将查询实体进行嵌入
    question_embeddings = call_embedding_api(query_entities)

    # 处理query_entities
    entities = process_question_kg(question_embeddings, embeddings)
    return entities

# 调用
# if __name__ == "__main__":
#     entities_file_path = "./download/raw_data/2wikimultihopqa/entity2id.txt"
#     embeddings_file_path = 'embeddings.json'
#     query_entities = ["Bernard B", "Comeback", "Cicero", "Part", "Ernesto"]
#
#     matched_entities = match_entities(entities_file_path, embeddings_file_path, query_entities)
#     print("Matched Entities:", matched_entities)
