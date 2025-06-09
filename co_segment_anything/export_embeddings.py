from neo4j import GraphDatabase
import pandas as pd
import numpy as np

def export_embeddings(uri, user, password, query, output_csv_path, ep=False):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run(query)
        rows = []
        for record in result:
            node_id = record["node_id"]
            embedding = record["embedding"]
            label = record.get("label", None)  # Only exists in MiniGrid

            if ep:
                task = record['task']
                rows.append({"id": node_id, 'task':task, "label": label, **{f"dim_{i}": v for i, v in enumerate(embedding)}})
            else:
                rows.append({"id": node_id, "label": label, **{f"dim_{i}": v for i, v in enumerate(embedding)}})


    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Exported to {output_csv_path}")

# Example usage:
# export_embeddings(
#     uri="bolt://localhost:7687",
#     user="neo4j",
#     password="minigrid123",
#     query="""
#     USE ltm
#     MATCH (o:ObjectConcept)
#              WHERE o.value is not NULL AND o.name is not NULL
#              RETURN id(o) AS node_id, o.name AS label, o.value AS embedding""",
#     output_csv_path="minigrid_embeddings.csv"
# )

# export_embeddings(
#     uri="bolt://localhost:7687",
#     user="neo4j",
#     password="rl123456",
#     query="""MATCH (o:ObjectConcept)
#              WHERE o.value is not NULL
#              RETURN id(o) AS node_id, o.value AS embedding""",
#     output_csv_path="create_embeddings.csv"
# )


export_embeddings(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="rl123456",
    query="""
     USE outcomesmall
    MATCH (e:Episode)
             WHERE e.set_embedding is not NULL
             RETURN id(e) AS node_id, e.set_embedding AS embedding, e.task as task""",
    output_csv_path="create_ep_embeddings.csv",
    ep=True
)

# export_embeddings(
#     uri="bolt://localhost:7687",
#     user="neo4j",
#     password="minigrid123",
#     query="""
#     USE ltm2
#     MATCH (e:Episode)
#              WHERE e.set_embedding is not NULL
#              RETURN id(e) AS node_id, e.set_embedding AS embedding, e.task as task""",
#     output_csv_path="minigrid_ep_embeddings.csv",
#     ep=True
# )
