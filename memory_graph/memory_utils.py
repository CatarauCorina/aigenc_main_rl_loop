import os
from graphdatascience import GraphDataScience
import uuid


class WorkingMemory:
    DATABASE_URL = os.environ["NEO4J_BOLT_URL"]
    NEO_USER = os.environ['NEO_USER']
    NEO_PASS = os.environ['NEO_PASS']

    def __init__(self, default_name='objectConcept'):
        self.gds = GraphDataScience(self.DATABASE_URL, auth=(self.NEO_USER, self.NEO_PASS))
        print(self.gds.version())
        self.project_name = default_name
        if self.check_if_graph_name_exists(default_name):
            self.project_name = default_name + str(uuid.uuid4())
        self.gds_init_project_catalog()
        #self.gds.set_database("concept_space")
        return

    def gds_init_project_catalog(self):
        self.gds.run_cypher(
            f"""
            CALL gds.graph.project(
            '{self.project_name}',
            {{
            ObjectConcept: {{
            properties: ['value']
            }}
            }},
            '*'
            );
            """
        )
        print(self.project_name)
        return

    def check_if_graph_name_exists(self, name):
        name_exists = self.gds.run_cypher(f"""
            RETURN gds.graph.exists('{name}') AS name_exists
        """)
        return name_exists['name_exists'][0]

    def create_state_graph(self):
        return

    def compute_effect(self):
        return

    def compute_silhouette_best_k(self):
        k_silhouette = self.gds.run_cypher(
            """
            WITH range(2,14) AS kcol
            UNWIND kcol AS k
                CALL gds.beta.kmeans.stream('objectConcept',
                {
                    nodeProperty: 'value',
                    computeSilhouette: true,
                    k: k
                }
            ) YIELD nodeId, communityId, silhouette
            WITH k, avg(silhouette) AS avgSilhouette
            RETURN k, avgSilhouette
            """
        )
        best_k = k_silhouette['k'][k_silhouette['avgSilhouette'].idxmax()]
        return best_k

    def compute_wm_clusters(self):
        best_k = cs_memory.compute_silhouette_best_k()
        k_clustering_result = self.gds.run_cypher(
            f"""
            CALL gds.beta.kmeans.stream('objectConcept', {{
            nodeProperty: 'value',
            k: {best_k},
            randomSeed: 42
            }})
            YIELD nodeId, communityId,distanceFromCentroid
            RETURN elementId(gds.util.asNode(nodeId)) AS id, communityId, distanceFromCentroid
            ORDER BY communityId, id ASC, distanceFromCentroid
            """
        )
        return k_clustering_result

    def compute_save_wm_clusters(self):
        best_k = cs_memory.compute_silhouette_best_k()
        k_clustering_result = self.gds.run_cypher(
            f"""
              CALL gds.beta.kmeans.write( 
                '{self.project_name}',
                {{
                    nodeProperty: 'value',
                    writeProperty: 'clusterVal',
                    k: {best_k}
                }}
            ) YIELD nodePropertiesWritten
               """
        )
        return k_clustering_result

    def compute_clusters_centroid(self):
        centroids = self.gds.run_cypher(
            """
                MATCH (u:ObjectConcept) WITH u.km13 AS cluster, u, range(0, 1000) AS ii 
                UNWIND ii AS i
                WITH cluster, i, avg(u.value[i]) AS avgVal
                ORDER BY cluster, i
                WITH cluster, collect(avgVal) AS avgEmbeddings
                MERGE (cl:Centroid{dimension: 1000, cluster: cluster})
                SET cl.embedding = avgEmbeddings
                RETURN cl;
            """
        )
        return centroids

    def add_to_memory(self):
        return


if __name__ == "__main__":
    cs_memory = WorkingMemory()
    clusters = cs_memory.compute_wm_clusters()
    print(clusters)
