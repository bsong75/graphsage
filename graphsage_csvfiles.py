import pandas as pd

def run_graphsage(self):
    """Run GraphSAGE for pest prediction using pre-computed structural and community features"""
    print("Starting GraphSAGE pest prediction pipeline")
    
    # Step 1: Add basic node properties
    print("Adding basic node properties...")
    self.gds.run_cypher("""MATCH (n) SET n.degree = size([(n)--() | 1])""")
    self.gds.run_cypher("""MATCH (e:Entity) SET e.entity_degree = size([(e)--() | 1])""")
    self.gds.run_cypher("""MATCH (t:TargetProxy) SET t.pest_value = t.value""")
    print("Basic node properties added successfully")
    
    # Step 2: Load pre-computed features from CSV files
    print("Loading pre-computed features from CSV files...")
    
    try:
        # Load structural features
        structural_df = pd.read_csv('structural.csv')
        print(f"Loaded structural features: {structural_df.shape}")
        print(f"Structural columns: {structural_df.columns.tolist()}")
        
        # Load community features  
        community_df = pd.read_csv('community.csv')
        print(f"Loaded community features: {community_df.shape}")
        print(f"Community columns: {community_df.columns.tolist()}")
        
        # Merge structural and community features
        if 'entity_id' in structural_df.columns and 'entity_id' in community_df.columns:
            combined_features = pd.merge(structural_df, community_df, on='entity_id', how='inner')
        else:
            # Assume first column is entity_id if not explicitly named
            structural_df.rename(columns={structural_df.columns[0]: 'entity_id'}, inplace=True)
            community_df.rename(columns={community_df.columns[0]: 'entity_id'}, inplace=True)
            combined_features = pd.merge(structural_df, community_df, on='entity_id', how='inner')
        
        print(f"Combined features shape: {combined_features.shape}")
        print(f"Combined feature columns: {combined_features.columns.tolist()}")
        
        # Get list of feature columns (excluding entity_id)
        feature_columns = [col for col in combined_features.columns if col != 'entity_id']
        print(f"Feature columns for GraphSAGE: {feature_columns}")
        
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        print("Please ensure 'structural.csv' and 'community.csv' exist in the current directory")
        return None
    
    # Step 3: Write features to Neo4j as node properties
    print("Writing pre-computed features to Neo4j...")
    
    # Convert DataFrame to dictionary for efficient lookups
    features_dict = combined_features.set_index('entity_id').to_dict('index')
    
    # Update nodes with features using batch processing
    batch_size = 1000
    entity_ids = list(features_dict.keys())
    
    for i in range(0, len(entity_ids), batch_size):
        batch_ids = entity_ids[i:i + batch_size]
        batch_data = {eid: features_dict[eid] for eid in batch_ids}
        
        # Create Cypher query to update nodes
        cypher_query = """
        UNWIND $batch_data AS row
        MATCH (e:Entity {id: row.entity_id})
        SET e += row.features
        """
        
        # Prepare batch data for Cypher
        batch_params = []
        for entity_id, features in batch_data.items():
            batch_params.append({
                'entity_id': entity_id,
                'features': features
            })
        
        self.gds.run_cypher(cypher_query, {'batch_data': batch_params})
    
    print("Pre-computed features written to Neo4j successfully")
    
    # Step 4: Create prediction projection with all properties
    print("Creating prediction projection with enhanced properties...")
    try:
        self.gds.graph.drop("pest_prediction")
        print("Dropped existing prediction graph")
    except:
        print("No existing prediction graph to drop")
        
    # Build nodeProperties list dynamically
    node_properties = ["degree", "pest_value", "entity_degree"] + feature_columns
    
    G_pred, _ = self.gds.graph.project(
        "pest_prediction",
        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"],
        {
            'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'},
            'SHIPPED_IN': {'orientation': 'NATURAL'},
            'IS_FROM': {'orientation': 'NATURAL'}, 
            'HAS_WEATHER': {'orientation': 'NATURAL'}
        },
        nodeProperties=node_properties
    )
    print("Enhanced prediction projection created successfully")
    
    # Step 5: Train GraphSAGE model with enhanced features
    print("Training GraphSAGE model with enhanced features...")
    
    # Use all available features for training
    training_features = ["degree", "entity_degree"] + feature_columns
    
    train_result = self.gds.beta.graphSage.train(
        G_pred,
        modelName="pest_predictor",
        featureProperties=training_features,
        projectedFeatureDimension=128,  # Increased dimension for more features
        randomSeed=42,
        epochs=20,
        batchSize=256,
        learningRate=0.01,
        sampleSizes=[25, 10]
    )
    
    print("GraphSAGE model training completed")
    print(f"Training metrics: {train_result}")
    
    # Step 6: Generate embeddings
    print("Generating entity embeddings...")
    
    # Build dynamic Cypher query for all features
    feature_selections = ", ".join([f"n.{col} as {col}" for col in feature_columns])
    
    cypher_query = f"""
        CALL gds.beta.graphSage.stream('pest_prediction', 
                                        {{modelName: 'pest_predictor' }}) 
        YIELD nodeId, embedding
        WITH nodeId, embedding
        MATCH (n) WHERE id(n) = nodeId AND 'Entity' IN labels(n)
        RETURN nodeId, n.id as entity_id, embedding,
               n.degree as degree,
               n.entity_degree as entity_degree,
               {feature_selections}
        """
    
    entity_embeddings = self.gds.run_cypher(cypher_query)
    
    # Step 7: Convert to DataFrame with all features
    print("Converting embeddings and features to DataFrame format...")
    embedding_df = pd.DataFrame(entity_embeddings['embedding'].tolist())
    embedding_df.columns = [f'graphsage_dim_{i}' for i in range(len(embedding_df.columns))]
    
    # Combine embeddings with computed properties
    all_feature_columns = ['degree', 'entity_degree'] + feature_columns
    
    final_df = pd.concat([
        entity_embeddings[['nodeId', 'entity_id'] + all_feature_columns],
        embedding_df
    ], axis=1)
    
    # Step 8: Get pest labels
    print("Retrieving pest labels for entities...")
    entity_labels = self.gds.run_cypher("""
        MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
        RETURN e.id as entity_id, 
            max(t.value) as has_pest_ever,
            avg(t.value) as pest_rate
        """)
    
    final_df = pd.merge(final_df, entity_labels, on='entity_id', how='left')
    
    # Step 9: Summary and output
    pest_count = final_df['has_pest_ever'].sum()
    total_entities = len(final_df)
    print(f"GraphSAGE analysis complete: {pest_count}/{total_entities} entities have pest history")
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Total feature columns: {len(all_feature_columns + embedding_df.columns.tolist())}")
    print(f"Structural features used: {[col for col in feature_columns if col in structural_df.columns]}")
    print(f"Community features used: {[col for col in feature_columns if col in community_df.columns]}")
    
    output_file = 'enhanced_graphsage_entity_features.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Enhanced GraphSAGE features saved to '{output_file}'")
    
    # # Cleanup
    # print("Cleaning up graph projections...")
    # try:
    #     self.gds.graph.drop("pest_prediction")
    #     print("Graph projections cleaned up successfully")
    # except:
    #     print("Some graph projections may still exist")
    
    # return final_df