def run_graphsage(self):
    """Run GraphSAGE for pest prediction with comprehensive node properties"""
    print("Starting GraphSAGE pest prediction pipeline")
    
    # Step 1: Add basic node properties
    print("Adding basic node properties...")
    self.gds.run_cypher("""MATCH (n) SET n.degree = size([(n)--() | 1])""")
    self.gds.run_cypher("""MATCH (e:Entity) SET e.entity_degree = size([(e)--() | 1])""")
    self.gds.run_cypher("""MATCH (t:TargetProxy) SET t.pest_value = t.value""")
    print("Basic node properties added successfully")
    
    # Step 2: Create graph projection for community detection and centrality
    print("Creating graph projection for property computation...")
    try:
        self.gds.graph.drop("property_computation")
        print("Dropped existing property computation graph")
    except:
        print("No existing property computation graph to drop")
    
    G_props, _ = self.gds.graph.project(
        "property_computation",
        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"],
        {
            'HAS_INSPECTION_RESULT': {'orientation': 'UNDIRECTED'},
            'SHIPPED_IN': {'orientation': 'UNDIRECTED'},
            'IS_FROM': {'orientation': 'UNDIRECTED'},
            'HAS_WEATHER': {'orientation': 'UNDIRECTED'}
        }
    )
    print("Property computation projection created successfully")
    
    # Step 3: Community Detection Algorithms
    print("Computing community detection properties...")
    
    # Louvain Community Detection
    print("Running Louvain community detection...")
    louvain_result = self.gds.louvain.mutate(
        G_props,
        mutateProperty="louvain_community",
        randomSeed=42,
        maxIterations=100,
        tolerance=0.0001
    )
    print(f"Louvain modularity: {louvain_result['modularity']}")
    
    # Leiden Community Detection  
    print("Running Leiden community detection...")
    leiden_result = self.gds.leiden.mutate(
        G_props,
        mutateProperty="leiden_community",
        randomSeed=42,
        maxIterations=100,
        tolerance=0.0001,
        gamma=1.0
    )
    print(f"Leiden modularity: {leiden_result['modularity']}")
    
    # Modularity Optimization
    print("Running Modularity Optimization...")
    modularity_result = self.gds.modularityOptimization.mutate(
        G_props,
        mutateProperty="modularity_community",
        randomSeed=42,
        maxIterations=100,
        tolerance=0.0001
    )
    print(f"Modularity optimization score: {modularity_result['modularity']}")
    
    # Step 4: Structural Centrality Algorithms
    print("Computing structural centrality properties...")
    
    # PageRank
    print("Computing PageRank centrality...")
    pagerank_result = self.gds.pageRank.mutate(
        G_props,
        mutateProperty="pagerank",
        dampingFactor=0.85,
        maxIterations=100,
        tolerance=0.0001
    )
    print(f"PageRank iterations: {pagerank_result['ranIterations']}")
    
    # Betweenness Centrality
    print("Computing Betweenness centrality...")
    betweenness_result = self.gds.betweenness.mutate(
        G_props,
        mutateProperty="betweenness"
    )
    
    # Closeness Centrality
    print("Computing Closeness centrality...")
    closeness_result = self.gds.closeness.mutate(
        G_props,
        mutateProperty="closeness"
    )
    
    # Eigenvector Centrality
    print("Computing Eigenvector centrality...")
    eigenvector_result = self.gds.eigenvector.mutate(
        G_props,
        mutateProperty="eigenvector",
        maxIterations=100,
        tolerance=0.0001
    )
    print(f"Eigenvector centrality iterations: {eigenvector_result['ranIterations']}")
    
    # Step 5: Write computed properties back to Neo4j
    print("Writing computed properties back to Neo4j...")
    
    # Write community properties
    self.gds.graph.nodeProperties.write(
        G_props,
        ["louvain_community", "leiden_community", "modularity_community"],
        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"]
    )
    
    # Write centrality properties
    self.gds.graph.nodeProperties.write(
        G_props,
        ["pagerank", "betweenness", "closeness", "eigenvector"],
        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"]
    )
    
    print("All properties written to Neo4j successfully")
    
    # Step 6: Create prediction projection with all properties
    print("Creating prediction projection with enhanced properties...")
    try:
        self.gds.graph.drop("pest_prediction")
        print("Dropped existing prediction graph")
    except:
        print("No existing prediction graph to drop")
        
    G_pred, _ = self.gds.graph.project(
        "pest_prediction",
        ["Entity", "TargetProxy", "Country", "Month", "CountryMonth"],
        {
            'HAS_INSPECTION_RESULT': {'orientation': 'NATURAL'},
            'SHIPPED_IN': {'orientation': 'NATURAL'},
            'IS_FROM': {'orientation': 'NATURAL'}, 
            'HAS_WEATHER': {'orientation': 'NATURAL'}
        },
        nodeProperties=[
            "degree", "pest_value", "entity_degree",
            "louvain_community", "leiden_community", "modularity_community",
            "pagerank", "betweenness", "closeness", "eigenvector"
        ]
    )
    print("Enhanced prediction projection created successfully")
    
    # Step 7: Train GraphSAGE model with enhanced features
    print("Training GraphSAGE model with enhanced features...")
    train_result = self.gds.beta.graphSage.train(
        G_pred,
        modelName="pest_predictor",
        featureProperties=[
            "degree", "entity_degree",
            "louvain_community", "leiden_community", "modularity_community",
            "pagerank", "betweenness", "closeness", "eigenvector"
        ],
        projectedFeatureDimension=128,  # Increased dimension for more features
        randomSeed=42,
        epochs=20,
        batchSize=256,
        learningRate=0.01,
        sampleSizes=[25, 10]
    )
    
    print("GraphSAGE model training completed")
    print(f"Training metrics: {train_result}")
    
    # Step 8: Generate embeddings
    print("Generating entity embeddings...")
    entity_embeddings = self.gds.run_cypher("""
        CALL gds.beta.graphSage.stream('pest_prediction', 
                                        {modelName: 'pest_predictor' }) 
        YIELD nodeId, embedding
        WITH nodeId, embedding
        MATCH (n) WHERE id(n) = nodeId AND 'Entity' IN labels(n)
        RETURN nodeId, n.id as entity_id, embedding,
               n.degree as degree,
               n.entity_degree as entity_degree,
               n.louvain_community as louvain_community,
               n.leiden_community as leiden_community,
               n.modularity_community as modularity_community,
               n.pagerank as pagerank,
               n.betweenness as betweenness,
               n.closeness as closeness,
               n.eigenvector as eigenvector
        """)
    
    # Step 9: Convert to DataFrame with all features
    print("Converting embeddings and features to DataFrame format...")
    embedding_df = pd.DataFrame(entity_embeddings['embedding'].tolist())
    embedding_df.columns = [f'graphsage_dim_{i}' for i in range(len(embedding_df.columns))]
    
    # Combine embeddings with computed properties
    feature_columns = [
        'degree', 'entity_degree', 'louvain_community', 'leiden_community',
        'modularity_community', 'pagerank', 'betweenness', 'closeness', 'eigenvector'
    ]
    
    final_df = pd.concat([
        entity_embeddings[['nodeId', 'entity_id'] + feature_columns],
        embedding_df
    ], axis=1)
    
    # Step 10: Get pest labels
    print("Retrieving pest labels for entities...")
    entity_labels = self.gds.run_cypher("""
        MATCH (e:Entity)-[:HAS_INSPECTION_RESULT]->(t:TargetProxy)
        RETURN e.id as entity_id, 
            max(t.value) as has_pest_ever,
            avg(t.value) as pest_rate
        """)
    
    final_df = pd.merge(final_df, entity_labels, on='entity_id', how='left')
    
    # Step 11: Summary and output
    pest_count = final_df['has_pest_ever'].sum()
    total_entities = len(final_df)
    print(f"GraphSAGE analysis complete: {pest_count}/{total_entities} entities have pest history")
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Feature columns: {final_df.columns.tolist()}")
    
    output_file = 'enhanced_graphsage_entity_features.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Enhanced GraphSAGE features saved to '{output_file}'")
    
    # # Cleanup
    # print("Cleaning up graph projections...")
    # try:
    #     self.gds.graph.drop("property_computation")
    #     self.gds.graph.drop("pest_prediction")
    #     print("Graph projections cleaned up successfully")
    # except:
    #     print("Some graph projections may still exist")
    
    # return final_df