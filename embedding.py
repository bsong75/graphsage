def extract_embedding_features(self):
    """Extract node embedding features using FastRP"""
    self.logger.info("Computing FastRP embeddings...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        # Run FastRP with configurable parameters
        result = self.gds.fastRP.stream(
            self.graph,
            embeddingDimension=64,
            iterationWeights=[0.8, 1, 1, 1],  # weights for 1-hop, 2-hop, 3-hop, 4-hop neighbors
            randomSeed=42
        )
        
        self.logger.info(f"FastRP completed. Generated embeddings for {len(result)} nodes")
        
        # Convert embeddings to DataFrame columns
        embedding_df = pd.DataFrame(result['embedding'].tolist())
        embedding_df.columns = [f'fastrp_dim_{i}' for i in range(len(embedding_df.columns))]
        
        # Combine with nodeId
        final_df = pd.concat([
            result[['nodeId']],
            embedding_df
        ], axis=1)
        
        # Merge with entity data
        entity_df = pd.merge(entity_df, final_df, on="nodeId")
        
        self.logger.info(f"FastRP embeddings extracted. Final shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"FastRP failed: {e}")
        # Create dummy embedding features if FastRP fails
        for i in range(64):
            entity_df[f'fastrp_dim_{i}'] = 0.0
    
    return entity_df

def extract_all_features(self):
    """Extract all feature types: structural, community, and embeddings"""
    self.logger.info("Starting comprehensive feature extraction...")
    
    # Get base entity DataFrame
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    # 1. Structural features
    self.logger.info("Extracting structural features...")
    structural_features = self.extract_structural_features()
    entity_df = pd.merge(entity_df, structural_features.drop('entity_id', axis=1), on="nodeId")
    
    # 2. Community features  
    self.logger.info("Extracting community features...")
    community_features = self.extract_community_features()
    entity_df = pd.merge(entity_df, community_features.drop('entity_id', axis=1), on="nodeId")
    
    # 3. Embedding features
    self.logger.info("Extracting embedding features...")
    embedding_features = self.extract_embedding_features()
    entity_df = pd.merge(entity_df, embedding_features.drop('entity_id', axis=1), on="nodeId")
    
    self.logger.info(f"All features extracted. Final shape: {entity_df.shape}")
    return entity_df


def extract_embedding_features(self, method='fastRP'):
    """Extract node embeddings using various methods"""
    self.logger.info(f"Computing {method} embeddings...")
    
    entity_df = self.gds.run_cypher('MATCH (n:Entity) RETURN id(n) as nodeId, n.id as entity_id')
    
    try:
        if method == 'fastRP':
            result = self.gds.fastRP.stream(
                self.graph,
                embeddingDimension=64,
                iterationWeights=[0.8, 1, 1, 1],
                randomSeed=42
            )
        elif method == 'node2vec':
            result = self.gds.node2vec.stream(
                self.graph,
                embeddingDimension=64,
                walkLength=10,
                walksPerNode=10,
                randomSeed=42
            )
        else:
            raise ValueError(f"Unknown embedding method: {method}")
        
        # Process embeddings
        embedding_df = pd.DataFrame(result['embedding'].tolist())
        embedding_df.columns = [f'{method}_dim_{i}' for i in range(len(embedding_df.columns))]
        
        final_df = pd.concat([result[['nodeId']], embedding_df], axis=1)
        entity_df = pd.merge(entity_df, final_df, on="nodeId")
        
        self.logger.info(f"{method} embeddings extracted. Shape: {entity_df.shape}")
        
    except Exception as e:
        self.logger.error(f"{method} failed: {e}")
        # Create dummy features
        for i in range(64):
            entity_df[f'{method}_dim_{i}'] = 0.0
    
    return entity_df