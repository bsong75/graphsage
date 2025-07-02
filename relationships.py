def create_temporal_relationships(self):
    """Create time-based relationships"""
    
    # Consecutive shipments by same entity
    self.gds.run_cypher("""
    MATCH (e:Entity)-[:SHIPPED_IN]->(m1:Month),
          (e:Entity)-[:SHIPPED_IN]->(m2:Month)
    WHERE m1.name <> m2.name
    MERGE (e)-[:SHIPPED_CONSECUTIVELY {
        from_month: m1.name, 
        to_month: m2.name
    }]->(e)
    """)
    
    # Seasonal patterns
    self.gds.run_cypher("""
    MATCH (e:Entity)-[:SHIPPED_IN]->(m:Month)
    WHERE m.name IN ['December', 'January', 'February']
    SET e.season = 'winter'
    """)
    
    # Connect entities that ship in same season
    self.gds.run_cypher("""
    MATCH (e1:Entity), (e2:Entity)
    WHERE e1.season = e2.season AND e1 <> e2
    MERGE (e1)-[:SAME_SEASON]->(e2)
    """)

def create_risk_relationships(self):
    """Create risk-based connections"""
    
    # High-risk entity clustering
    self.gds.run_cypher("""
    MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy {value: 1})
    SET e.risk_level = 'high'
    """)
    
    # Connect high-risk entities from same country
    self.gds.run_cypher("""
    MATCH (e1:Entity {risk_level: 'high'})-[:IS_FROM]->(c:Country),
          (e2:Entity {risk_level: 'high'})-[:IS_FROM]->(c)
    WHERE e1 <> e2
    MERGE (e1)-[:HIGH_RISK_SAME_COUNTRY]->(e2)
    """)
    
    # Connect entities with similar pest rates
    self.gds.run_cypher("""
    MATCH (e1:Entity), (e2:Entity)
    WHERE e1 <> e2
    WITH e1, e2,
         [(e1)-[:HAS_INSPECTION_RESULT]->(t1:TargetProxy) | t1.value] as pests1,
         [(e2)-[:HAS_INSPECTION_RESULT]->(t2:TargetProxy) | t2.value] as pests2
    WITH e1, e2, 
         reduce(sum1 = 0, p IN pests1 | sum1 + p) * 1.0 / size(pests1) as rate1,
         reduce(sum2 = 0, p IN pests2 | sum2 + p) * 1.0 / size(pests2) as rate2
    WHERE abs(rate1 - rate2) < 0.1
    MERGE (e1)-[:SIMILAR_PEST_RATE {rate_diff: abs(rate1 - rate2)}]->(e2)
    """)

def create_cooccurrence_relationships(self):
    """Create relationships based on co-occurrence patterns"""
    
    # Entities inspected together (same country + month)
    self.gds.run_cypher("""
    MATCH (e1:Entity), (e2:Entity),
          (i1:Inspection)-[:FOR_ENTITY]->(e1),
          (i2:Inspection)-[:FOR_ENTITY]->(e2)
    WHERE e1 <> e2 
      AND i1.country_code = i2.country_code 
      AND i1.month = i2.month
    MERGE (e1)-[:INSPECTED_TOGETHER {
        country: i1.country_code,
        month: i1.month
    }]->(e2)
    """)
    
    # Entities from same country in different months
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:IS_FROM]->(c:Country),
          (e2:Entity)-[:IS_FROM]->(c),
          (e1)-[:SHIPPED_IN]->(m1:Month),
          (e2)-[:SHIPPED_IN]->(m2:Month)
    WHERE e1 <> e2 AND m1 <> m2
    MERGE (e1)-[:SAME_COUNTRY_DIFF_TIME]->(e2)
    """)

def create_volume_relationships(self):
    """Create relationships based on inspection volumes"""
    
    # High-volume entities (lots of inspections)
    self.gds.run_cypher("""
    MATCH (e:Entity)
    WITH e, size([(e)<-[:FOR_ENTITY]-(i:Inspection) | i]) as inspection_count
    WHERE inspection_count > 10
    SET e.volume_level = 'high'
    """)
    
    # Connect high-volume entities
    self.gds.run_cypher("""
    MATCH (e1:Entity {volume_level: 'high'}),
          (e2:Entity {volume_level: 'high'})
    WHERE e1 <> e2
    MERGE (e1)-[:HIGH_VOLUME_PEER]->(e2)
    """)
    
    # Connect entities with similar examination patterns
    self.gds.run_cypher("""
    MATCH (e1:Entity), (e2:Entity)
    WHERE e1 <> e2
    WITH e1, e2,
         [(e1)<-[:FOR_ENTITY]-(i1:Inspection) | i1.exams_30d] as exams1,
         [(e2)<-[:FOR_ENTITY]-(i2:Inspection) | i2.exams_30d] as exams2
    WITH e1, e2,
         reduce(sum1 = 0, e IN exams1 | sum1 + e) / size(exams1) as avg_exams1,
         reduce(sum2 = 0, e IN exams2 | sum2 + e) / size(exams2) as avg_exams2
    WHERE abs(avg_exams1 - avg_exams2) < 5
    MERGE (e1)-[:SIMILAR_EXAM_VOLUME]->(e2)
    """)

def create_geographic_relationships(self):
    """Create geographic relationships"""
    
    # Define regional clusters
    regions = {
        'North_America': ['US', 'CA', 'MX'],
        'Europe': ['DE', 'FR', 'IT', 'ES', 'UK'],
        'Asia': ['CN', 'JP', 'KR', 'IN'],
        'South_America': ['BR', 'AR', 'CL']
    }
    
    for region, countries in regions.items():
        country_list = "', '".join(countries)
        self.gds.run_cypher(f"""
        MATCH (c:Country)
        WHERE c.code IN ['{country_list}']
        SET c.region = '{region}'
        """)
    
    # Connect entities from same region
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[:IS_FROM]->(c1:Country),
          (e2:Entity)-[:IS_FROM]->(c2:Country)
    WHERE e1 <> e2 AND c1.region = c2.region AND c1 <> c2
    MERGE (e1)-[:SAME_REGION]->(e2)
    """)


def create_enhanced_projection(self):
    """Create projection with all relationship types"""
    
    # Add all relationships first
    self.create_temporal_relationships()
    self.create_risk_relationships() 
    self.create_cooccurrence_relationships()
    self.create_volume_relationships()
    self.create_geographic_relationships()
    
    # Drop existing projection
    try:
        self.gds.graph.drop("pest_graph_enhanced")
    except:
        pass
    
    # Create enhanced projection
    self.enhanced_graph, _ = self.gds.graph.project(
        "pest_graph_enhanced",
        ["Country", "Month", "Entity", "TargetProxy", "CountryMonth"],
        [
            # Original relationships
            'SHIPPED_IN', 'IS_FROM', 'HAS_WEATHER', 'HAS_INSPECTION_RESULT',
            # New relationships
            'SHIPPED_CONSECUTIVELY', 'SAME_SEASON', 'HIGH_RISK_SAME_COUNTRY',
            'SIMILAR_PEST_RATE', 'INSPECTED_TOGETHER', 'SAME_COUNTRY_DIFF_TIME',
            'HIGH_VOLUME_PEER', 'SIMILAR_EXAM_VOLUME', 'SAME_REGION'
        ]
    )
    
    self.logger.info(f"Enhanced graph created with {self.enhanced_graph.node_count()} nodes and {self.enhanced_graph.relationship_count()} relationships")
    
    return self.enhanced_graph

def extract_enhanced_structural_features(self):
    """Extract features from enhanced graph"""
    
    # Create enhanced projection
    enhanced_graph = self.create_enhanced_projection()
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    # Centrality measures on enhanced graph
    centrality_algorithms = ['pageRank', 'betweenness', 'closeness', 'eigenvector', 'degree']
    
    for algo in centrality_algorithms:
        method = getattr(self.gds, algo)
        result = method.stream(enhanced_graph).rename(columns={'score': f'enhanced_{algo}'})
        entity_df = pd.merge(entity_df, result, on="nodeId")
    
    # Additional measures
    triangles = self.gds.triangleCount.stream(enhanced_graph)
    entity_df = pd.merge(entity_df, triangles[['nodeId', 'triangleCount']].rename(columns={'triangleCount': 'enhanced_triangles'}), on="nodeId")
    
    clustering = self.gds.localClusteringCoefficient.stream(enhanced_graph)
    entity_df = pd.merge(entity_df, clustering[['nodeId', 'localClusteringCoefficient']].rename(columns={'localClusteringCoefficient': 'enhanced_clustering'}), on="nodeId")
    
    return entity_df