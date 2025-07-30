# -*- coding: utf-8 -*-

import asyncio

import ollama
import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Load tables to dataframes
INPUT_DIR = "G:/code/graphrag-local-ollama/download/raw_data/2wikimultihopqa/rag_1"
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
# Read entities
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
# read nodes table to get community and degree data
output_dir = f"{INPUT_DIR}/output/20240918-222550/artifacts"
entity_df = pd.read_parquet(f"{output_dir}/{ENTITY_TABLE}.parquet")
entity_embedding_df = pd.read_parquet(f"{output_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")

entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

# load description embeddings to an in-memory lancedb vectorstore
# to connect to a remote db, specify url and port values.
description_embedding_store = LanceDBVectorStore(
    collection_name="entity_description_embeddings",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)
entity_description_embeddings = store_entity_semantic_embeddings(
    entities=entities, vectorstore=description_embedding_store
)

# print(f"Entity count: {len(entity_df)}")
entity_df.head()
relationship_df = pd.read_parquet(f"{output_dir}/{RELATIONSHIP_TABLE}.parquet")
relationships = read_indexer_relationships(relationship_df)

# print(f"Relationship count: {len(relationship_df)}")
# Read relationships
relationship_df.head()
covariate_df = pd.read_parquet(f"{output_dir}/{COVARIATE_TABLE}.parquet")

claims = read_indexer_covariates(covariate_df)

# print(f"Claim records: {len(claims)}")
covariates = {"claims": claims}
# Read community reports
report_df = pd.read_parquet(f"{output_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

# print(f"Report records: {len(report_df)}")
report_df.head()

# Read text units
text_unit_df = pd.read_parquet(f"{output_dir}/{TEXT_UNIT_TABLE}.parquet")
text_units = read_indexer_text_units(text_unit_df)

# print(f"Text unit records: {len(text_unit_df)}")
text_unit_df.head()

api_key = "ollama"
llm_model = "qwen2.5:7b-instruct"
embedding_model = "nomic-embed-text"

llm = ChatOpenAI(
    api_key=api_key,
    model=llm_model,
    api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
    max_retries=20,
    api_base='http://localhost:11434/v1'
)

token_encoder = tiktoken.get_encoding("cl100k_base")

text_embedder = OpenAIEmbedding(
    api_key=api_key,
    api_base='http://localhost:11434/api',
    model=embedding_model,
)
# text_embedder=ollama.embed(model=embedding_model)

# Create local search context builder
context_builder = LocalSearchMixedContext(
    community_reports=reports,
    text_units=text_units,
    entities=entities,
    relationships=relationships,
    covariates=covariates,
    entity_text_embeddings=description_embedding_store,
    # embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    embedding_vectorstore_key=EntityVectorStoreKey.TITLE,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
    text_embedder=text_embedder,
    token_encoder=token_encoder,
)

# Create local search engine
local_context_params = {
    "max_tokens": 4096,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
}

llm_params = {
    "max_tokens": 512,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
    "temperature": 0.0,
    "top_p": 0,
}

search_engine = LocalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=token_encoder,
    llm_params=llm_params,
    context_builder_params=local_context_params,
    response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
)

# Run local search on sample queries
async def run_local_search(question):
    # return await search_engine.asearch(question)

    async for result in search_engine.asearch(question):
        print(result.response)  # 每次有结果时立即输出

import sys
sys.stdout.reconfigure(encoding='utf-8')
if __name__ == "__main__":
    INPUT_DIR=sys.argv[1]
    question = sys.argv[2]
    # INPUT_DIR = "F:/code/HToG-RAG/2wiki"
    # question = "What is the capital of France?"
    # 运行异步搜索并立即输出结果
    asyncio.run(run_local_search(question))