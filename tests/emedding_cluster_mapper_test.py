import numpy as np
import pandas as pd

from src.preprocessing.embedding_cluster_mapper import EmbeddingClusterMapper


class TestEmbeddingClusterMapper:
    def test_cluster_mapping(self):
        # Sample metadata
        test_samples_length = 80
        train_samples_length = 200

        metadata = pd.DataFrame(
            {
                "file_path": [f"audio{i}.wav" for i in range(test_samples_length)],
                "label": np.random.choice([0, 1], size=test_samples_length),
            }
        )

        # Sample embeddings
        train_embeddings = np.random.rand(train_samples_length, 128)

        test_embeddings = np.random.rand(test_samples_length, 128)

        # Initialize the mapper
        mapper = EmbeddingClusterMapper(K_list=[2, 4, 8, 10], random_state=42)

        # Get cluster info
        silhouette_scores, inertias = mapper.get_info_for_each_cluster(train_embeddings)
        print("Silhouette Scores:", silhouette_scores)
        print("Inertias:", inertias)

        # Train the model
        n_clusters = 4
        model, pca = mapper.train(train_embeddings, n_clusters=n_clusters)

        # save and load the model
        mapper.save_model(model, pca, file_name="test_cluster_model.pkl")
        loaded_model, loaded_pca = mapper.load_model(file_name="test_cluster_model.pkl")

        # Transform the embeddings
        transformed_metadata = mapper.transform(metadata, test_embeddings, pca, model)
        transformed_metadata_with_loaded_models = mapper.transform(metadata, test_embeddings, loaded_pca, loaded_model)

        assert transformed_metadata.shape[0] == test_embeddings.shape[0]
        assert transformed_metadata_with_loaded_models.shape[0] == test_embeddings.shape[0]
        assert pd.Series.equals(transformed_metadata["cluster"], transformed_metadata_with_loaded_models["cluster"])
        print("Cluster assignments:", transformed_metadata["cluster"].values)
        print("Test passed.")


TestEmbeddingClusterMapper().test_cluster_mapping()
