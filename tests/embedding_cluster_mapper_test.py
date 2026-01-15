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

        np.random.seed(42)
        # Sample embeddings
        train_embeddings = np.random.rand(train_samples_length, 128)

        test_embeddings = np.random.rand(test_samples_length, 128)

        # Initialize the mapper
        mapper = EmbeddingClusterMapper(min_cluster_size=5, min_samples=2, random_state=42)

        # Train the model
        n_components = 10
        model, pca = mapper.train(train_embeddings, n_components=n_components)

        # save and load the model
        mapper.save_model(model, pca, file_name="test_cluster_model.pkl")
        loaded_model, loaded_pca = mapper.load_model(file_name="test_cluster_model.pkl")

        # Transform the embeddings
        clusters = mapper.transform(test_embeddings, pca, model)
        print(clusters.shape, clusters)
        transformed_metadata = mapper.assign_clusters_to_metadata(metadata, clusters)
        clusters_loaded = mapper.transform(test_embeddings, loaded_pca, loaded_model)
        transformed_metadata_with_loaded_models = mapper.assign_clusters_to_metadata(metadata, clusters_loaded)

        print("Shape of transformed metadata:", transformed_metadata.shape)
        print("Shape of transformed metadata with loaded models:", transformed_metadata_with_loaded_models.shape)
        # assert transformed_metadata.shape[0] == test_embeddings.shape[0]
        # assert transformed_metadata_with_loaded_models.shape[0] == test_embeddings.shape[0]
        assert transformed_metadata.shape == transformed_metadata_with_loaded_models.shape
        assert pd.Series.equals(transformed_metadata["cluster"], transformed_metadata_with_loaded_models["cluster"])
        print("Cluster assignments:", transformed_metadata["cluster"].values)
        print("Test passed.")


TestEmbeddingClusterMapper().test_cluster_mapping()
