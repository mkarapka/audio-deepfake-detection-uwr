from src.pipelines.set_audio_ids_pipeline import SetAudioIDsPipeline

if __name__ == "__main__":
    output_file = "feature_extracted_audio_ids"
    pipeline = SetAudioIDsPipeline(output_file=output_file)
    pipeline.run()
