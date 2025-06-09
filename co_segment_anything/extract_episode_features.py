from neo4j import GraphDatabase
import pandas as pd

# Neo4j connection
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "rl123456"))

def get_episode_features(tx):
    query = """
    use outcomesmall
   MATCH (e:Episode)-[:HAS_STATE]->(s:StateT)
WITH e, collect(s) AS states
UNWIND states AS s

// Count unique states
WITH e, s, count(DISTINCT s) AS num_states

// Count affordances linked from states
OPTIONAL MATCH (s)-[:influences]->(a:Affordance)
WITH e, s, a, num_states,
     count(DISTINCT a) AS num_affordances

// Count has_object edges
OPTIONAL MATCH (s)-[:has_object]->(o:ObjectConcept)
WITH e, s, a, o, num_states, num_affordances,
     count(DISTINCT o) AS num_objects

// Count outcome edges
OPTIONAL MATCH (a)-[r:outcome]->(s2:StateT)
WITH e, num_states, num_affordances, num_objects,
     count(DISTINCT r) AS num_outcomes

// Count influences edges
OPTIONAL MATCH (s)-[r2:influences]->(a2:Affordance)
WITH e, num_states, num_affordances, num_objects, num_outcomes,
     count(DISTINCT r2) AS num_influences

// Get episode depth from timestep
OPTIONAL MATCH (e)-[:HAS_STATE]->(s:StateT)
WITH e, num_states, num_affordances, num_objects, num_outcomes, num_influences,
     max(s.timestep) AS episode_depth

RETURN elementId(e) AS episode_id,
       e.task AS task,
       e.env AS env,
       num_states + num_affordances + num_objects AS num_nodes,
       num_states,
       num_affordances,
       num_objects,
       num_influences,
       num_outcomes,
       episode_depth

    """
    result = tx.run(query)
    return [dict(record) for record in result]  # ✅ CONSUME HERE

with driver.session() as session:
    data = session.read_transaction(get_episode_features)



    # Save to CSV
df = pd.DataFrame(data)
df.to_csv("struct_features_ep.csv", index=False)
print("✅ Saved: struct_features_ep.csv")

driver.close()
