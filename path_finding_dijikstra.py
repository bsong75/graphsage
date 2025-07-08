"""
dijkstra_features.py

Simple Dijkstra pathfinding feature extraction for Neo4j graph database.
"""

import pandas as pd


def create_weighted_relationships(gds):
    """Create weighted relationships for better Dijkstra results"""
    
    # Weight relationships based on risk and similarity
    gds.run_cypher("""
    MATCH (e1:Entity)-[r:SIMILAR_PEST_RATE]->(e2:Entity)
    SET r.weight = 1.0 + r.rate_diff * 10
    """)
    
    gds.run_cypher("""
    MATCH (e1:Entity)-[r:HIGH_RISK_SAME_COUNTRY]->(e2:Entity)
    SET r.weight = 0.5
    """)
    
    gds.run_cypher("""
    MATCH (e1:Entity)-[r:SAME_REGION]->(e2:Entity)
    SET r.weight = 2.0
    """)
    
    # Default weight for other relationships
    gds.run_cypher("""
    MATCH (e1:Entity)-[r]->(e2:Entity)
    WHERE r.weight IS NULL
    SET r.weight = 1.0
    """)


def extract_dijkstra_features(gds, graph_name="pest_graph_enhanced", max_sources=20):
    """
    Extract Dijkstra shortest path features from high-risk entities.
    
    Args:
        gds: Neo4j Graph Data Science client
        graph_name: Name of the graph projection
        max_sources: Maximum number of high-risk entities to use as sources
        
    Returns:
        DataFrame with Dijkstra-based features
    """
    
    # Get entity dataframe
    entity_df = gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    if entity_df.empty:
        print("No entities found in graph")
        return pd.DataFrame()
    
    # Get high-risk entities as source nodes
    high_risk_entities = gds.run_cypher(f"""
    MATCH (e:Entity {{risk_level: 'high'}})
    RETURN id(e) as nodeId, e.id as entity_id
    LIMIT {max_sources}
    """)
    
    if high_risk_entities.empty:
        print("No high-risk entities found for Dijkstra analysis")
        return pd.DataFrame({'nodeId': entity_df['nodeId']})
    
    print(f"Found {len(high_risk_entities)} high-risk entities as sources")
    
    dijkstra_features = []
    
    for _, source_entity in high_risk_entities.iterrows():
        source_id = source_entity['nodeId']
        entity_id = source_entity['entity_id']
        
        try:
            # Run Dijkstra from this high-risk entity
            dijkstra_result = gds.shortestPath.dijkstra.stream(
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
            print(f"Dijkstra failed for source {source_id}: {e}")
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
        
        print(f"Generated {dijkstra_agg.shape[1]} Dijkstra features for {len(dijkstra_agg)} entities")
        
        return dijkstra_agg
    else:
        print("No Dijkstra features generated")
        return pd.DataFrame({'nodeId': entity_df['nodeId']})


def run_dijkstra_analysis(gds, graph_name="pest_graph_enhanced"):
    """
    Run complete Dijkstra analysis with weighted relationships.
    
    Args:
        gds: Neo4j Graph Data Science client
        graph_name: Name of the graph projection
        
    Returns:
        DataFrame with Dijkstra features
    """
    
    print("Starting Dijkstra pathfinding analysis")
    
    # Create weighted relationships
    print("Creating weighted relationships...")
    create_weighted_relationships(gds)
    
    # Extract Dijkstra features
    print("Extracting Dijkstra features...")
    dijkstra_features = extract_dijkstra_features(gds, graph_name)
    
    print("Dijkstra analysis complete")
    
    return dijkstra_features


def get_dijkstra_feature_summary(features_df):
    """
    Get summary of Dijkstra features.
    
    Args:
        features_df: DataFrame with Dijkstra features
        
    Returns:
        Dictionary with summary statistics
    """
    
    feature_cols = [col for col in features_df.columns if col != 'nodeId']
    
    summary = {
        'total_entities': len(features_df),
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'feature_stats': features_df[feature_cols].describe()
    }
    
    return summary