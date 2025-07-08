import pandas as pd

# Add these methods to your PestDataAnalyzer class

def create_weighted_relationships(self):
    """Create weighted relationships for better pathfinding results"""
    
    # Weight relationships based on risk and similarity
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:SIMILAR_PEST_RATE]->(e2:Entity)
    SET r.weight = 1.0 + r.rate_diff * 10
    """)
    
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:HIGH_RISK_SAME_COUNTRY]->(e2:Entity)
    SET r.weight = 0.5
    """)
    
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:SAME_REGION]->(e2:Entity)
    SET r.weight = 2.0
    """)
    
    # Default weight for other relationships
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r]->(e2:Entity)
    WHERE r.weight IS NULL
    SET r.weight = 1.0
    """)

def extract_dijkstra_features(self, graph_name="pest_graph_enhanced", max_sources=20):
    """Extract Dijkstra shortest path features from high-risk entities"""
    
    # Get entity dataframe
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    if entity_df.empty:
        return pd.DataFrame()
    
    # Get high-risk entities as source nodes
    high_risk_entities = self.gds.run_cypher(f"""
    MATCH (e:Entity {{risk_level: 'high'}})
    RETURN id(e) as nodeId, e.id as entity_id
    LIMIT {max_sources}
    """)
    
    if high_risk_entities.empty:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})
    
    dijkstra_features = []
    
    for _, source_entity in high_risk_entities.iterrows():
        source_id = source_entity['nodeId']
        entity_id = source_entity['entity_id']
        
        try:
            # Run Dijkstra from this high-risk entity
            dijkstra_result = self.gds.shortestPath.dijkstra.stream(
                graph_name,
                sourceNode=source_id,
                relationshipWeightProperty='weight'
            )
            
            # Process results for feature extraction
            for _, row in dijkstra_result.iterrows():
                target_id = row['targetNode']
                distance = row['totalCost']
                
                dijkstra_features.append({
                    'nodeId': target_id,
                    f'dijkstra_dist_from_{entity_id}': distance
                })
        
        except Exception as e:
            continue
    
    # Convert to DataFrame and aggregate
    if dijkstra_features:
        dijkstra_df = pd.DataFrame(dijkstra_features)
        
        # Get distance columns
        distance_cols = [col for col in dijkstra_df.columns if col.startswith('dijkstra_dist_from_')]
        
        # Calculate summary statistics per entity
        dijkstra_agg = dijkstra_df.groupby('nodeId')[distance_cols].agg(['min', 'max', 'mean', 'std']).reset_index()
        
        # Flatten column names
        dijkstra_agg.columns = ['nodeId'] + [f"dijkstra_{col[1]}_{col[0].split('_')[-1]}" for col in dijkstra_agg.columns[1:]]
        
        # Add count of reachable high-risk entities
        dijkstra_agg['dijkstra_reachable_high_risk_count'] = dijkstra_df.groupby('nodeId').size().reset_index(drop=True)
        
        # Fill NaN values
        dijkstra_agg = dijkstra_agg.fillna(float('inf'))
        
        return dijkstra_agg
    else:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})

def extract_dfs_features(self, graph_name="pest_graph_enhanced", sample_size=10, max_depth=5):
    """Extract DFS traversal features from sample entities"""
    
    # Get entity dataframe
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    if entity_df.empty:
        return pd.DataFrame()
    
    # Get sample of entities as starting points for DFS
    actual_sample_size = min(sample_size, len(entity_df))
    sample_entities = entity_df.sample(n=actual_sample_size)
    
    dfs_features = []
    
    for _, source_entity in sample_entities.iterrows():
        source_id = source_entity['nodeId']
        
        try:
            # Run DFS traversal
            dfs_result = self.gds.dfs.stream(
                graph_name,
                sourceNode=source_id,
                maxDepth=max_depth
            )
            
            # Track DFS metrics for each node
            dfs_metrics = {}
            
            for _, row in dfs_result.iterrows():
                node_id = row['nodeId']
                depth = row['depth']
                
                if node_id not in dfs_metrics:
                    dfs_metrics[node_id] = {
                        'dfs_reachable_count': 1,
                        'dfs_min_depth': depth,
                        'dfs_max_depth': depth,
                        'dfs_depth_sum': depth,
                        'dfs_visit_count': 1
                    }
                else:
                    dfs_metrics[node_id]['dfs_min_depth'] = min(dfs_metrics[node_id]['dfs_min_depth'], depth)
                    dfs_metrics[node_id]['dfs_max_depth'] = max(dfs_metrics[node_id]['dfs_max_depth'], depth)
                    dfs_metrics[node_id]['dfs_depth_sum'] += depth
                    dfs_metrics[node_id]['dfs_visit_count'] += 1
            
            # Convert to features list
            for node_id, metrics in dfs_metrics.items():
                metrics['nodeId'] = node_id
                metrics['dfs_avg_depth'] = metrics['dfs_depth_sum'] / metrics['dfs_visit_count']
                dfs_features.append(metrics)
        
        except Exception as e:
            continue
    
    # Aggregate DFS features
    if dfs_features:
        dfs_df = pd.DataFrame(dfs_features)
        
        # Group by node and aggregate across different source nodes
        dfs_agg = dfs_df.groupby('nodeId').agg({
            'dfs_reachable_count': 'sum',
            'dfs_min_depth': 'min',
            'dfs_max_depth': 'max',
            'dfs_avg_depth': 'mean',
            'dfs_visit_count': 'sum'
        }).reset_index()
        
        # Calculate reachability ratio
        dfs_agg['dfs_reachability_ratio'] = dfs_agg['dfs_reachable_count'] / len(sample_entities)
        
        return dfs_agg
    else:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})

def run_pathfinding_analysis(self, graph_name="pest_graph_enhanced"):
    """Run complete pathfinding analysis with Dijkstra and DFS"""
    
    # Create weighted relationships
    self.create_weighted_relationships()
    
    # Get entity dataframe
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    if entity_df.empty:
        return pd.DataFrame()
    
    # Extract Dijkstra features
    dijkstra_features = self.extract_dijkstra_features(graph_name)
    
    # Extract DFS features
    dfs_features = self.extract_dfs_features(graph_name)
    
    # Merge all features
    pathfinding_features = entity_df.copy()
    pathfinding_features = pd.merge(pathfinding_features, dijkstra_features, on="nodeId", how="left")
    pathfinding_features = pd.merge(pathfinding_features, dfs_features, on="nodeId", how="left")
    
    # Fill NaN values
    pathfinding_features = pathfinding_features.fillna(0)
    
    return pathfinding_features