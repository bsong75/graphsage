import pandas as pd
import numpy as np

def extract_dijkstra_features(self):
    """Extract shortest path features using Dijkstra algorithm"""
    self.logger.info("Computing Dijkstra shortest path features...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        # Single-source shortest path from high-risk entities
        high_risk_entities = self.gds.run_cypher("""
        MATCH (e:Entity {risk_level: 'high'})
        RETURN id(e) as nodeId
        LIMIT 10
        """)
        
        dijkstra_features = []
        
        for _, row in high_risk_entities.iterrows():
            source_id = row['nodeId']
            
            # Run Dijkstra from this high-risk entity
            result = self.gds.shortestPath.dijkstra.stream(
                self.enhanced_graph,
                sourceNode=source_id,
                relationshipWeightProperty='weight'  # You may need to add weights
            )
            
            # Calculate distance statistics for each target entity
            distance_stats = result.groupby('targetNodeId').agg({
                'totalCost': ['min', 'mean', 'max'],
                'nodeIds': 'count'
            }).reset_index()
            
            distance_stats.columns = ['nodeId', f'dijkstra_min_dist_from_{source_id}', 
                                    f'dijkstra_avg_dist_from_{source_id}', 
                                    f'dijkstra_max_dist_from_{source_id}',
                                    f'dijkstra_path_count_from_{source_id}']
            
            dijkstra_features.append(distance_stats)
        
        # Merge all Dijkstra features
        for features in dijkstra_features:
            entity_df = pd.merge(entity_df, features, on="nodeId", how="left")
        
        # All-pairs shortest path statistics (for smaller subgraphs)
        all_pairs_result = self.gds.allShortestPaths.dijkstra.stream(
            self.enhanced_graph,
            relationshipWeightProperty='weight'
        )
        
        # Aggregate statistics per entity
        path_stats = all_pairs_result.groupby('sourceNodeId').agg({
            'totalCost': ['min', 'mean', 'max', 'std'],
            'targetNodeId': 'count'
        }).reset_index()
        
        path_stats.columns = ['nodeId', 'dijkstra_min_total_cost', 'dijkstra_avg_total_cost',
                             'dijkstra_max_total_cost', 'dijkstra_std_total_cost', 
                             'dijkstra_reachable_nodes']
        
        entity_df = pd.merge(entity_df, path_stats, on="nodeId", how="left")
        
        self.logger.info(f"Dijkstra features extracted. Shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"Dijkstra feature extraction failed: {e}")
        # Create dummy features
        dummy_cols = ['dijkstra_min_total_cost', 'dijkstra_avg_total_cost', 
                     'dijkstra_max_total_cost', 'dijkstra_std_total_cost', 
                     'dijkstra_reachable_nodes']
        for col in dummy_cols:
            entity_df[col] = 0.0
    
    return entity_df

def extract_dfs_features(self):
    """Extract depth-first search features"""
    self.logger.info("Computing DFS traversal features...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        # DFS from each entity to measure reachability patterns
        dfs_features = []
        
        # Get sample of entities for DFS (to avoid computation explosion)
        sample_entities = entity_df.sample(n=min(50, len(entity_df)), random_state=42)
        
        for _, row in sample_entities.iterrows():
            source_id = row['nodeId']
            
            # Run DFS traversal
            result = self.gds.dfs.stream(
                self.enhanced_graph,
                sourceNode=source_id,
                maxDepth=5  # Limit depth to avoid infinite traversals
            )
            
            if not result.empty:
                # Calculate DFS statistics
                dfs_stats = {
                    'nodeId': source_id,
                    'dfs_total_nodes_reached': len(result),
                    'dfs_max_depth': result['path'].apply(len).max() if 'path' in result.columns else 0,
                    'dfs_avg_depth': result['path'].apply(len).mean() if 'path' in result.columns else 0,
                    'dfs_unique_relationships': result['path'].apply(lambda x: len(set(x)) if isinstance(x, list) else 0).sum()
                }
                
                dfs_features.append(dfs_stats)
        
        if dfs_features:
            dfs_df = pd.DataFrame(dfs_features)
            entity_df = pd.merge(entity_df, dfs_df, on="nodeId", how="left")
        
        # Calculate incoming DFS statistics (how often this entity is reached)
        incoming_stats = self.gds.run_cypher("""
        MATCH (source:Entity), (target:Entity)
        WHERE source <> target
        CALL gds.dfs.stream($graph_name, {sourceNode: id(source), maxDepth: 3})
        YIELD nodeId
        WHERE nodeId = id(target)
        RETURN id(target) as nodeId, count(*) as dfs_incoming_count
        """, {'graph_name': self.enhanced_graph.name()})
        
        entity_df = pd.merge(entity_df, incoming_stats, on="nodeId", how="left")
        
        self.logger.info(f"DFS features extracted. Shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"DFS feature extraction failed: {e}")
        # Create dummy features
        dummy_cols = ['dfs_total_nodes_reached', 'dfs_max_depth', 'dfs_avg_depth', 
                     'dfs_unique_relationships', 'dfs_incoming_count']
        for col in dummy_cols:
            entity_df[col] = 0.0
    
    return entity_df

def extract_bfs_features(self):
    """Extract breadth-first search features"""
    self.logger.info("Computing BFS traversal features...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        # BFS from high-risk entities to measure spread patterns
        high_risk_entities = self.gds.run_cypher("""
        MATCH (e:Entity {risk_level: 'high'})
        RETURN id(e) as nodeId
        LIMIT 20
        """)
        
        bfs_features = []
        
        for _, row in high_risk_entities.iterrows():
            source_id = row['nodeId']
            
            # Run BFS traversal
            result = self.gds.bfs.stream(
                self.enhanced_graph,
                sourceNode=source_id,
                maxDepth=4
            )
            
            if not result.empty:
                # Calculate BFS statistics for each depth level
                depth_stats = result.groupby('path').size().reset_index(name='nodes_at_depth')
                depth_stats['depth'] = depth_stats['path'].apply(len) - 1
                
                bfs_stats = {
                    'nodeId': source_id,
                    'bfs_total_reachable': len(result),
                    'bfs_max_depth': depth_stats['depth'].max(),
                    'bfs_nodes_at_depth_1': depth_stats[depth_stats['depth'] == 1]['nodes_at_depth'].sum(),
                    'bfs_nodes_at_depth_2': depth_stats[depth_stats['depth'] == 2]['nodes_at_depth'].sum(),
                    'bfs_nodes_at_depth_3': depth_stats[depth_stats['depth'] == 3]['nodes_at_depth'].sum(),
                    'bfs_branching_factor': depth_stats['nodes_at_depth'].mean()
                }
                
                bfs_features.append(bfs_stats)
        
        if bfs_features:
            bfs_df = pd.DataFrame(bfs_features)
            
            # For non-high-risk entities, calculate their distance from high-risk entities
            all_bfs_distances = []
            
            for _, row in high_risk_entities.iterrows():
                source_id = row['nodeId']
                
                distances = self.gds.bfs.stream(
                    self.enhanced_graph,
                    sourceNode=source_id,
                    maxDepth=4
                )
                
                if not distances.empty:
                    distances['distance_from_high_risk'] = distances['path'].apply(len) - 1
                    distances['source_high_risk'] = source_id
                    all_bfs_distances.append(distances[['nodeId', 'distance_from_high_risk', 'source_high_risk']])
            
            if all_bfs_distances:
                distance_df = pd.concat(all_bfs_distances, ignore_index=True)
                
                # Calculate minimum distance from any high-risk entity
                min_distances = distance_df.groupby('nodeId')['distance_from_high_risk'].min().reset_index()
                min_distances.columns = ['nodeId', 'bfs_min_distance_from_high_risk']
                
                entity_df = pd.merge(entity_df, min_distances, on="nodeId", how="left")
                
                # Calculate average distance from high-risk entities
                avg_distances = distance_df.groupby('nodeId')['distance_from_high_risk'].mean().reset_index()
                avg_distances.columns = ['nodeId', 'bfs_avg_distance_from_high_risk']
                
                entity_df = pd.merge(entity_df, avg_distances, on="nodeId", how="left")
        
        # Calculate BFS connectivity patterns
        connectivity_stats = self.gds.run_cypher("""
        MATCH (e:Entity)
        CALL gds.bfs.stream($graph_name, {sourceNode: id(e), maxDepth: 2})
        YIELD nodeId
        WITH e, count(nodeId) as two_hop_neighbors
        RETURN id(e) as nodeId, two_hop_neighbors as bfs_two_hop_connectivity
        """, {'graph_name': self.enhanced_graph.name()})
        
        entity_df = pd.merge(entity_df, connectivity_stats, on="nodeId", how="left")
        
        self.logger.info(f"BFS features extracted. Shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"BFS feature extraction failed: {e}")
        # Create dummy features
        dummy_cols = ['bfs_min_distance_from_high_risk', 'bfs_avg_distance_from_high_risk', 
                     'bfs_two_hop_connectivity']
        for col in dummy_cols:
            entity_df[col] = 0.0
    
    return entity_df

def extract_pathfinding_features(self):
    """Extract all path-finding algorithm features"""
    self.logger.info("Starting path-finding feature extraction...")
    
    # Ensure enhanced graph exists
    if not hasattr(self, 'enhanced_graph') or self.enhanced_graph is None:
        self.create_enhanced_projection()
    
    # Get base entity DataFrame
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    # Extract Dijkstra features
    self.logger.info("Extracting Dijkstra features...")
    dijkstra_features = self.extract_dijkstra_features()
    entity_df = pd.merge(entity_df, dijkstra_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # Extract DFS features
    self.logger.info("Extracting DFS features...")
    dfs_features = self.extract_dfs_features()
    entity_df = pd.merge(entity_df, dfs_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # Extract BFS features
    self.logger.info("Extracting BFS features...")
    bfs_features = self.extract_bfs_features()
    entity_df = pd.merge(entity_df, bfs_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # Fill NaN values with appropriate defaults
    numeric_columns = entity_df.select_dtypes(include=[np.number]).columns
    entity_df[numeric_columns] = entity_df[numeric_columns].fillna(0)
    
    self.logger.info(f"All path-finding features extracted. Final shape: {entity_df.shape}")
    return entity_df

def extract_advanced_path_features(self):
    """Extract advanced path-based features"""
    self.logger.info("Computing advanced path features...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        # Shortest path between entities of different risk levels
        risk_path_stats = self.gds.run_cypher("""
        MATCH (high:Entity {risk_level: 'high'}), (low:Entity)
        WHERE NOT low.risk_level = 'high' AND high <> low
        CALL gds.shortestPath.dijkstra.stream($graph_name, {
            sourceNode: id(high),
            targetNode: id(low)
        })
        YIELD sourceNodeId, targetNodeId, totalCost, path
        RETURN targetNodeId as nodeId, 
               min(totalCost) as min_path_to_high_risk,
               avg(totalCost) as avg_path_to_high_risk,
               count(*) as path_count_to_high_risk
        """, {'graph_name': self.enhanced_graph.name()})
        
        entity_df = pd.merge(entity_df, risk_path_stats, on="nodeId", how="left")
        
        # Path diversity (number of different paths between entities)
        path_diversity = self.gds.run_cypher("""
        MATCH (e:Entity)
        CALL gds.allShortestPaths.dijkstra.stream($graph_name, {
            sourceNode: id(e)
        })
        YIELD sourceNodeId, targetNodeId, path
        WITH sourceNodeId, count(DISTINCT targetNodeId) as unique_targets,
             avg(size(path)) as avg_path_length
        RETURN sourceNodeId as nodeId, unique_targets as path_diversity,
               avg_path_length as avg_outgoing_path_length
        """, {'graph_name': self.enhanced_graph.name()})
        
        entity_df = pd.merge(entity_df, path_diversity, on="nodeId", how="left")
        
        # Centrality based on path-finding
        betweenness_centrality = self.gds.betweenness.stream(self.enhanced_graph)
        entity_df = pd.merge(entity_df, 
                           betweenness_centrality[['nodeId', 'score']].rename(columns={'score': 'path_betweenness'}), 
                           on="nodeId", how="left")
        
        self.logger.info(f"Advanced path features extracted. Shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"Advanced path feature extraction failed: {e}")
        # Create dummy features
        dummy_cols = ['min_path_to_high_risk', 'avg_path_to_high_risk', 'path_count_to_high_risk',
                     'path_diversity', 'avg_outgoing_path_length', 'path_betweenness']
        for col in dummy_cols:
            entity_df[col] = 0.0
    
    return entity_df

def extract_all_features_with_pathfinding(self):
    """Extract all feature types including path-finding algorithms"""
    self.logger.info("Starting comprehensive feature extraction with path-finding...")
    
    # Get base entity DataFrame
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    # 1. Structural features
    self.logger.info("Extracting structural features...")
    structural_features = self.extract_enhanced_structural_features()
    entity_df = pd.merge(entity_df, structural_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # 2. Community features  
    self.logger.info("Extracting community features...")
    community_features = self.extract_community_features()
    entity_df = pd.merge(entity_df, community_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # 3. Embedding features
    self.logger.info("Extracting embedding features...")
    embedding_features = self.extract_embedding_features()
    entity_df = pd.merge(entity_df, embedding_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # 4. Path-finding features
    self.logger.info("Extracting path-finding features...")
    pathfinding_features = self.extract_pathfinding_features()
    entity_df = pd.merge(entity_df, pathfinding_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # 5. Advanced path features
    self.logger.info("Extracting advanced path features...")
    advanced_path_features = self.extract_advanced_path_features()
    entity_df = pd.merge(entity_df, advanced_path_features.drop('entity_id', axis=1), on="nodeId", how="left")
    
    # Fill any remaining NaN values
    numeric_columns = entity_df.select_dtypes(include=[np.number]).columns
    entity_df[numeric_columns] = entity_df[numeric_columns].fillna(0)
    
    self.logger.info(f"All features including path-finding extracted. Final shape: {entity_df.shape}")
    return entity_df