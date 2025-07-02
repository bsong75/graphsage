def create_basic_cooccurrence_relationships(self):
    """Create fundamental co-occurrence relationships"""
    
    # 1. Entities shipping in same month
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:SHIPPED_IN]->(m:Month)<-[:SHIPPED_IN]-(e2:Entity)
    WHERE e1 <> e2
    MERGE (e1)-[:SAME_MONTH]->(e2)
    """)
    
    # 2. Entities from same country  
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:IS_FROM]->(c:Country)<-[:IS_FROM]-(e2:Entity)
    WHERE e1 <> e2
    MERGE (e1)-[:SAME_COUNTRY]->(e2)
    """)
    
    # 3. Entities with same weather conditions (same CountryMonth)
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:HAS_WEATHER]->(cm:CountryMonth)<-[:HAS_WEATHER]-(e2:Entity)
    WHERE e1 <> e2
    MERGE (e1)-[:SAME_WEATHER]->(e2)
    """)
    
    # 4. Entities with same inspection result
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)<-[:HAS_INSPECTION_RESULT]-(e2:Entity)
    WHERE e1 <> e2
    MERGE (e1)-[:SAME_RESULT]->(e2)
    """)

# Current graph (limited relationships):
# Entity → Country (1 connection)
# Entity → Month (1 connection)

# # With basic co-occurrence:
# Entity → Entity (via same month) (potentially 100s of connections)
# Entity → Entity (via same country) (potentially 100s of connections)
# # This dramatically increases the entity-to-entity connectivity, which will give you much higher variance in:

# PageRank (entities in common months/countries become more important)
# Betweenness (some entities bridge different month/country clusters)
# Clustering (entities form tight groups by shared characteristics)

def create_basic_enhanced_projection(self):
    """Create projection with basic co-occurrence relationships"""
    
    # Add basic relationships
    self.create_basic_cooccurrence_relationships()
    
    # Create projection
    try:
        self.gds.graph.drop("pest_graph_basic_enhanced")
    except:
        pass
    
    self.enhanced_graph, _ = self.gds.graph.project(
        "pest_graph_basic_enhanced",
        ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
        [
            # Original relationships
            'SHIPPED_IN', 'IS_FROM', 'HAS_WEATHER', 'HAS_INSPECTION_RESULT',
            # Basic co-occurrence relationships  
            'SAME_MONTH', 'SAME_COUNTRY', 'SAME_WEATHER', 'SAME_RESULT'
        ]
    )
    
    return self.enhanced_graph