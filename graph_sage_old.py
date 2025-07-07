    # Create constraints for better performance
    constraints = [
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT country_code_unique IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE", 
        "CREATE CONSTRAINT month_name_unique IF NOT EXISTS FOR (m:Month) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT target_value_unique IF NOT EXISTS FOR (t:TargetProxy) REQUIRE t.value IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            gds.run_cypher(constraint)
        except Exception as e:
            pass  # Constraint may already exist
    


    ######### GraphSAGE for Pest Prediction  ###########
    ####################################################
    def run_graphsage_prediction(gds, G):
        print("=== GraphSAGE Pest Prediction ===")
        # Step 1: Add node properties for GraphSAGE to use
        print("Step 1: Adding node properties...")
        # Add degree centrality as a property for all nodes
        gds.run_cypher("""
        MATCH (n)
        SET n.degree = size([(n)--() | 1])
        """)
        # Add specific properties for different node types
        gds.run_cypher("""
        MATCH (e:Entity)
        SET e.entity_degree = size([(e)--() | 1])
        """)
        gds.run_cypher("""
        MATCH (t:TargetProxy)
        SET t.pest_value = t.value
        """)
        print("Added node properties")
        
        # Step 2: Create projection with node properties
        print("Step 2: Creating projection with properties...")
        try:
            gds.graph.drop("pest_prediction")
        except:
            pass
            
        G_pred, _ = gds.graph.project(
                                        "pest_prediction",
                                        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"],
                                        {
                                            'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'},
                                            'SHIPPED_IN': {'orientation': 'NATURAL'},
                                            'IS_FROM': {'orientation': 'NATURAL'}, 
                                            'HAS_WEATHER': {'orientation': 'NATURAL'}
                                        },
                                        nodeProperties=["degree", "pest_value", "entity_degree"]
                                    )
        
        # Step 3: Train GraphSAGE model with properties
        print("Step 3: Training GraphSAGE model...")
        train_result = gds.beta.graphSage.train(
            G_pred,
            modelName="pest_predictor",
            featureProperties=["degree"],  # Use degree as the main feature
            projectedFeatureDimension=64,
            randomSeed=42,
            epochs=20,
            batchSize=256,
            learningRate=0.01,
            sampleSizes=[25, 10]  # Neighbor sampling: 25 from 1-hop, 10 from 2-hop
        )
        
        print("Training metrics:")
        print(train_result)
        
        # Step 4: Generate embeddings for all nodes
        print("Step 4: Generating node embeddings...")
        embeddings = gds.beta.graphSage.stream(
                                                G_pred,
                                                modelName="pest_predictor"
                                            )
        
        print("Embeddings shape:", embeddings.shape)
        print("Sample embeddings:")
        print(embeddings.head())
        
        # Step 5: Get embeddings for entities specifically
        print("Step 5: Getting entity embeddings...")
        entity_embeddings = gds.run_cypher("""
                                            CALL gds.beta.graphSage.stream('pest_prediction', 
                                                                            {modelName: 'pest_predictor' }) 
                                            YIELD nodeId, embedding
                                            WITH nodeId, embedding
                                            MATCH (n) WHERE id(n) = nodeId AND 'Entity' IN labels(n)
                                            RETURN nodeId, n.id as entity_id, embedding
                                            """)
        
        # Step 6: Convert embeddings to DataFrame for ML
        print("Step 6: Converting to ML-ready format...")
        # Extract embedding dimensions into separate columns
        embedding_df = pd.DataFrame(entity_embeddings['embedding'].tolist())
        embedding_df.columns = [f'graphsage_dim_{i}' for i in range(len(embedding_df.columns))]
        
        # Combine with entity info
        final_df = pd.concat([
            entity_embeddings[['nodeId', 'entity_id']],
            embedding_df
        ], axis=1)
        
        print("Final DataFrame shape:", final_df.shape)
        print("Sample of final DataFrame:")
        print(final_df.head())
        
        # Step 7: Get actual pest labels for entities (for supervised learning)
        print("Step 7: Getting pest labels for entities...")
        entity_labels = gds.run_cypher("""
                                        MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
                                        RETURN e.id as entity_id, 
                                            max(t.value) as has_pest_ever,  // 1 if entity ever had pest
                                            avg(t.value) as pest_rate       // proportion of pest inspections
                                        """)
        
        # Merge labels with embeddings
        final_df = pd.merge(final_df, entity_labels, on='entity_id', how='left')
        
        print("Final DataFrame with labels:")
        print(final_df.head())
        print(f"Entities with pest history: {final_df['has_pest_ever'].sum()}")
        print(f"Total entities: {len(final_df)}")
        
        # Step 8: Save for ML use
        final_df.to_csv('graphsage_entity_features.csv', index=False)
        print("Saved GraphSAGE features to 'graphsage_entity_features.csv'")
        
        # Cleanup

        return final_df

    # gds.run_cypher("CALL gds.model.drop('pest_predictor')")
    # gds.graph.drop("pest_prediction")
    def clean_up(gds):
        try:
            # For models: Use the correct API
            gds.run_cypher("CALL gds.model.drop('pest_predictor')")
            print("Model 'pest_predictor' dropped successfully")
        except Exception as e:
            print(f"Model does not exist: {e}")
        
        try:
            # For graphs: This should work with string
            gds.graph.drop("pest_prediction") 
            print("Graph 'pest_prediction' dropped successfully")
        except Exception as e:
            print(f"Graph does not exist: {e}")

    clean_up(gds)
    final_df = run_graphsage_prediction(gds, G)
    clean_up(gds)
    print("Final Dataframe \n", final_df)
