from neo4j import GraphDatabase, basic_auth

# 创建Neo4j驱动实例
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"

driver = GraphDatabase.driver(uri, auth=(username, password))

# 定义查询参数
start_entity_name = "Polish-Russian War"
end_entity_name = "Xawery Żuławski"

# 在会话中执行Cypher查询
with driver.session() as session:
    result = session.run(
        """
        MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name})
        MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity))
        RETURN p
        """,
        start_entity_name=start_entity_name,
        end_entity_name=end_entity_name
    )
    print(result)
    # 处理查询结果
    for record in result:
        print(record["p"])

# 关闭驱动连接
driver.close()
