def extract_pathfinding_features(self):
    """Extract path-based features using Dijkstra, DFS, and BFS algorithms"""
    
    # Ensure enhanced projection exists
    enhanced_graph = self.create_enhanced_projection()
    
    # Get entity dataframe
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    # Extract features from each pathfinding algorithm
    dijkstra_features = self.extract_dijkstra_features(enhanced_graph, entity_df)
    dfs_features = self.extract_dfs_features(enhanced_graph, entity_df)
    bfs_features = self.extract_bfs_features(enhanced_graph, entity_df)
    
    # Merge all features
    path_features = entity_df.copy()
    path_features = pd.merge(path_features, dijkstra_features, on="nodeId", how="left")
    path_features = pd.merge(path_features, dfs_features, on="nodeId", how="left")
    path_features = pd.merge(path_features, bfs_features, on="nodeId", how="left")
    
    self.logger.info(f"Extracted path-finding features for {len(path_features)} entities")
    
    return path_features

def extract_dijkstra_features(self, graph, entity_df):
    """Extract Dijkstra shortest path features"""
    
    # Get high-risk entities as source nodes
    high_risk_entities = self.gds.run_cypher("""
    MATCH (e:Entity {risk_level: 'high'})
    RETURN id(e) as nodeId, e.id as entity_id
    """)
    
    dijkstra_features = []
    
    for _, source_entity in high_risk_entities.iterrows():
        source_id = source_entity['nodeId']
        
        # Run Dijkstra from this high-risk entity
        dijkstra_result = self.gds.shortestPath.dijkstra.stream(
            graph,
            sourceNode=source_id,
            relationshipWeightProperty='weight'  # Use default weight of 1.0
        )
        
        # Process results for feature extraction
        for _, row in dijkstra_result.iterrows():
            target_id = row['targetNode']
            distance = row['totalCost']
            path_length = len(row['path']) if 'path' in row else 0
            
            dijkstra_features.append({
                'nodeId': target_id,
                f'dijkstra_dist_from_highrisk_{source_entity["entity_id"]}': distance,
                f'dijkstra_path_len_from_highrisk_{source_entity["entity_id"]}': path_length
            })
    
    # Aggregate features per entity
    dijkstra_df = pd.DataFrame(dijkstra_features)
    
    if not dijkstra_df.empty:
        # Calculate summary statistics
        dijkstra_agg = dijkstra_df.groupby('nodeId').agg({
            col: ['min', 'max', 'mean', 'std'] 
            for col in dijkstra_df.columns if col.startswith('dijkstra_')
        }).reset_index()
        
        # Flatten column names
        dijkstra_agg.columns = ['nodeId'] + [f"{col[0]}_{col[1]}" for col in dijkstra_agg.columns[1:]]
        
        # Fill NaN values for entities not reached
        dijkstra_agg = dijkstra_agg.fillna(float('inf'))
        
        return dijkstra_agg
    else:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})

def extract_dfs_features(self, graph, entity_df):
    """Extract DFS traversal features"""
    
    # Get sample of entities as starting points for DFS
    sample_entities = entity_df.sample(n=min(10, len(entity_df)))
    
    dfs_features = []
    
    for _, source_entity in sample_entities.iterrows():
        source_id = source_entity['nodeId']
        
        # Run DFS traversal
        dfs_result = self.gds.dfs.stream(
            graph,
            sourceNode=source_id,
            maxDepth=5  # Limit depth to avoid excessive computation
        )
        
        # Track DFS metrics for each node
        dfs_metrics = {}
        
        for _, row in dfs_result.iterrows():
            node_id = row['nodeId']
            depth = row['depth']
            
            if node_id not in dfs_metrics:
                dfs_metrics[node_id] = {
                    'dfs_reachable_from_sample': 1,
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
        
        # Convert to features
        for node_id, metrics in dfs_metrics.items():
            metrics['nodeId'] = node_id
            metrics['dfs_avg_depth'] = metrics['dfs_depth_sum'] / metrics['dfs_visit_count']
            dfs_features.append(metrics)
    
    # Aggregate DFS features
    if dfs_features:
        dfs_df = pd.DataFrame(dfs_features)
        
        # Group by node and aggregate across different source nodes
        dfs_agg = dfs_df.groupby('nodeId').agg({
            'dfs_reachable_from_sample': 'sum',
            'dfs_min_depth': 'min',
            'dfs_max_depth': 'max',
            'dfs_avg_depth': 'mean',
            'dfs_visit_count': 'sum'
        }).reset_index()
        
        # Calculate reachability ratio
        dfs_agg['dfs_reachability_ratio'] = dfs_agg['dfs_reachable_from_sample'] / len(sample_entities)
        
        return dfs_agg
    else:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})

def extract_bfs_features(self, graph, entity_df):
    """Extract BFS traversal features"""
    
    # Get entities from different countries as BFS sources
    country_entities = self.gds.run_cypher("""
    MATCH (e:Entity)-[:IS_FROM]->(c:Country)
    WITH c.code as country, collect(id(e)) as entities
    RETURN country, entities[0] as sample_entity_id
    LIMIT 10
    """)
    
    bfs_features = []
    
    for _, country_row in country_entities.iterrows():
        source_id = country_row['sample_entity_id']
        country = country_row['country']
        
        # Run BFS from this country representative
        bfs_result = self.gds.bfs.stream(
            graph,
            sourceNode=source_id,
            maxDepth=4
        )
        
        # Process BFS results
        for _, row in bfs_result.iterrows():
            node_id = row['nodeId']
            depth = row['depth']
            
            bfs_features.append({
                'nodeId': node_id,
                f'bfs_depth_from_{country}': depth,
                f'bfs_reachable_from_{country}': 1
            })
    
    # Convert to DataFrame and aggregate
    if bfs_features:
        bfs_df = pd.DataFrame(bfs_features)
        
        # Create pivot table for country-specific features
        bfs_pivot = bfs_df.pivot_table(
            index='nodeId',
            columns=[col for col in bfs_df.columns if col.startswith('bfs_depth_from_')],
            values=[col for col in bfs_df.columns if col.startswith('bfs_reachable_from_')],
            aggfunc='min',
            fill_value=float('inf')
        )
        
        # Flatten column names
        bfs_pivot.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in bfs_pivot.columns]
        bfs_pivot = bfs_pivot.reset_index()
        
        # Calculate summary statistics
        depth_cols = [col for col in bfs_pivot.columns if 'depth' in col]
        reachable_cols = [col for col in bfs_pivot.columns if 'reachable' in col]
        
        if depth_cols:
            bfs_pivot['bfs_avg_depth_across_countries'] = bfs_pivot[depth_cols].mean(axis=1)
            bfs_pivot['bfs_min_depth_across_countries'] = bfs_pivot[depth_cols].min(axis=1)
            bfs_pivot['bfs_max_depth_across_countries'] = bfs_pivot[depth_cols].max(axis=1)
        
        if reachable_cols:
            bfs_pivot['bfs_reachable_countries_count'] = (bfs_pivot[reachable_cols] == 1).sum(axis=1)
        
        return bfs_pivot
    else:
        return pd.DataFrame({'nodeId': entity_df['nodeId']})

def create_weighted_relationships(self):
    """Create weighted relationships for better Dijkstra results"""
    
    # Weight relationships based on risk and similarity
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:SIMILAR_PEST_RATE]->(e2:Entity)
    SET r.weight = 1.0 + r.rate_diff * 10
    """)
    
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:HIGH_RISK_SAME_COUNTRY]->(e2:Entity)
    SET r.weight = 0.5  // Lower weight for high-risk connections
    """)
    
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r:SAME_REGION]->(e2:Entity)
    SET r.weight = 2.0  // Higher weight for regional connections
    """)
    
    # Default weight for other relationships
    self.gds.run_cypher("""
    MATCH (e1:Entity)-[r]->(e2:Entity)
    WHERE r.weight IS NULL
    SET r.weight = 1.0
    """)

def extract_path_pattern_features(self, graph):
    """Extract features based on specific path patterns"""
    
    # Find paths through high-risk entities
    high_risk_paths = self.gds.run_cypher("""
    MATCH path = (e1:Entity)-[*2..4]->(e2:Entity)
    WHERE any(node in nodes(path) WHERE node.risk_level = 'high')
      AND e1 <> e2
    RETURN id(e1) as source_id, id(e2) as target_id, length(path) as path_length,
           size([node in nodes(path) WHERE node.risk_level = 'high']) as high_risk_nodes_in_path
    """)
    
    # Find paths through multiple countries
    multi_country_paths = self.gds.run_cypher("""
    MATCH path = (e1:Entity)-[*2..4]->(e2:Entity)
    WHERE e1 <> e2
    WITH path, e1, e2,
         [node in nodes(path) WHERE node:Entity | 
          [(node)-[:IS_FROM]->(c:Country) | c.code][0]] as countries_in_path
    WHERE size(apoc.coll.toSet(countries_in_path)) > 1
    RETURN id(e1) as source_id, id(e2) as target_id, length(path) as path_length,
           size(apoc.coll.toSet(countries_in_path)) as unique_countries_in_path
    """)
    
    # Aggregate path pattern features
    path_features = []
    
    # Process high-risk path features
    if not high_risk_paths.empty:
        high_risk_agg = high_risk_paths.groupby('source_id').agg({
            'path_length': ['min', 'max', 'mean'],
            'high_risk_nodes_in_path': ['max', 'mean'],
            'target_id': 'count'
        }).reset_index()
        
        high_risk_agg.columns = ['nodeId'] + [f"high_risk_path_{col[0]}_{col[1]}" for col in high_risk_agg.columns[1:]]
        high_risk_agg = high_risk_agg.rename(columns={'high_risk_path_target_id_count': 'high_risk_paths_count'})
        
        path_features.append(high_risk_agg)
    
    # Process multi-country path features
    if not multi_country_paths.empty:
        multi_country_agg = multi_country_paths.groupby('source_id').agg({
            'path_length': ['min', 'max', 'mean'],
            'unique_countries_in_path': ['max', 'mean'],
            'target_id': 'count'
        }).reset_index()
        
        multi_country_agg.columns = ['nodeId'] + [f"multi_country_path_{col[0]}_{col[1]}" for col in multi_country_agg.columns[1:]]
        multi_country_agg = multi_country_agg.rename(columns={'multi_country_path_target_id_count': 'multi_country_paths_count'})
        
        path_features.append(multi_country_agg)
    
    # Combine all path pattern features
    if path_features:
        combined_features = path_features[0]
        for i in range(1, len(path_features)):
            combined_features = pd.merge(combined_features, path_features[i], on="nodeId", how="outer")
        
        return combined_features.fillna(0)
    else:
        return pd.DataFrame()

def run_complete_pathfinding_analysis(self):
    """Run complete pathfinding feature extraction"""
    
    # Create weighted relationships for better pathfinding
    self.create_weighted_relationships()
    
    # Extract main pathfinding features
    pathfinding_features = self.extract_pathfinding_features()
    
    # Extract path pattern features
    enhanced_graph = self.create_enhanced_projection()
    path_pattern_features = self.extract_path_pattern_features(enhanced_graph)
    
    # Combine all features
    if not path_pattern_features.empty:
        complete_features = pd.merge(pathfinding_features, path_pattern_features, on="nodeId", how="left")
    else:
        complete_features = pathfinding_features
    
    # Fill NaN values
    complete_features = complete_features.fillna(0)
    
    self.logger.info(f"Complete pathfinding analysis generated {complete_features.shape[1]} features for {len(complete_features)} entities")
    
    return complete_features