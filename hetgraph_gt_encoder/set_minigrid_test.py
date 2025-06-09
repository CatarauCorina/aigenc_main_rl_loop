# SETLE-Based Graph Analysis for MiniGrid (Neo4j)
# ================================================

from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Neo4j Connection ---
def connect_neo4j(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))

# --- Fetch Affordance Transitions ---
def fetch_affordance_transitions(driver) -> pd.DataFrame:
    query = """
    MATCH (e:Episode)-[r1:has_state]->(s1:State)-[:influences]->(a:Affordance)-[:outcome]->(s2:State)<-[r2:has_state]-(e)
    WHERE r2.time = r1.time + 1
    OPTIONAL MATCH (s1)-[:has_object]->(o:Object)
    RETURN 
      e.id AS episode_id,
      r1.time AS timestep,
      collect(DISTINCT o.clip_label) AS visible_objects,
      a.name AS affordance_name,
      r2.time AS resulting_timestep,
      e.successful_outcome AS success
    ORDER BY episode_id, timestep
    """
    with driver.session() as session:
        result = session.run(query)
        data = [record.data() for record in result]
    return pd.DataFrame(data)

# --- Compute Affordance Success Rates ---
def compute_affordance_success(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(['affordance_name', 'success']).size().unstack(fill_value=0)
    grouped['success_rate'] = grouped.get(True, 0) / (grouped.get(True, 0) + grouped.get(False, 0))
    return grouped.sort_values(by='success_rate', ascending=False)

# --- Common Object-Affordance Pairs in Successful Episodes ---
def common_successful_pairs(df: pd.DataFrame) -> pd.DataFrame:
    success_df = df[df['success'] == True]
    expanded_rows = []
    for _, row in success_df.iterrows():
        for obj in row['visible_objects']:
            expanded_rows.append({'object': obj, 'affordance': row['affordance_name']})
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df.value_counts().reset_index(name='count').sort_values(by='count', ascending=False)

# --- Trace Object Trajectories Across Time ---
def object_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df.explode('visible_objects')
    return exploded.groupby(['episode_id', 'visible_objects'])['timestep'].apply(list).reset_index()

# --- Visualisation Example ---
def plot_affordance_success(aff_df: pd.DataFrame):
    aff_df = aff_df.reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=aff_df, x='affordance_name', y='success_rate')
    plt.xticks(rotation=45, ha='right')
    plt.title('Affordance Success Rates')
    plt.tight_layout()
    plt.show()
