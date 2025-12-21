import torch
import polars as pl

class AudioSegmentation:
    def __init__(self, overlap = 2, max_duration = 4):
        self.overlap_sec = overlap
        self.chunk_sec = max_duration
    
    def get_relevant_samples(self, wave_samples):
        samples = wave_samples.get_all_samples()
        return [
            samples.data,
            samples.sample_rate
        ]
        
    def devide_audio_into_slices(self, waveform, sr):
        chunk_samples = int(self.chunk_sec * sr)
        stride_samples = int((self.chunk_sec - self.overlap_sec) * sr)
        
        chunks = torch.split(waveform, chunk_samples, dim=-1, stride=stride_samples)
        starting_points = [i * stride_samples / sr for i in range(len(chunks))]
        
        return chunks, starting_points
    
    def transform(self, data_set):
        new_ds = {
            "key_id": [],
            "wave": [],
            "duration": [],
            "starting_point": [],
        }
        
        sample_rates = {
            "key_id": [],
            "sample_rate": []
        }
        for record in data_set:
            # print(record['wav'].get_all_samples())

            waveform, sr = self.get_relevant_samples(record['wav'])
            key_id = record['__key__']
            
            chunks, st_points = self.devide_audio_into_chunks(waveform, sr)
            for ch, stp in zip(chunks, st_points):
                new_ds["key_id"].append(key_id)
                new_ds['wave'].append(ch)
                new_ds['duration'].append(ch.shape[-1] / sr)
                new_ds['starting_point'].append(stp)
            
            
            sample_rates["key_id"].append(key_id)
            sample_rates["sample_rate"].append(sr)
            
        df_new_ds = pl.DataFrame(new_ds)
        df_sample_rates = pl.DataFrame(sample_rates)
        return df_new_ds, df_sample_rates