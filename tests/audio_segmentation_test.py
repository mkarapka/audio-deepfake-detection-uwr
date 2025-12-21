from datasets import load_dataset, Audio
from src.preprocessing.audio_segmentation import AudioSegmentation

class TestAudioSegmentation:
    def test_transform(self):
        dataset = load_dataset("wqz995/AUDETER", "mls-tts-bark")
        
        dataset = dataset['dev'].cast_column("wav", Audio())
        audio_segmenter = AudioSegmentation(overlap=1, max_duration=2)
        segmented_ds, sample_rates = audio_segmenter.transform(dataset)
        
        print(segmented_ds[0])
        print(sample_rates[0])
        
        
TestAudioSegmentation().test_transform()
