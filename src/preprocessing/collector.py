import logging
from pathlib import Path

import pandas as pd
import numpy as np
from src.common.constants import Constants as consts
from src.preprocessing.base_preprocessor import BasePreprocessor

logger = logging.getLogger("audio_deepfake.collector")


class Collector(BasePreprocessor):
    def __init__(self, save_file_name: str):
        super().__init__(class_name=__class__.__name__)
        self.data_dir = consts.data_dir / "collected_data"
        if self.data_dir.exists() is False:
            self.data_dir.mkdir(parents=True, exist_ok=True)

        if save_file_name is not None:
            self.save_file_path = self.data_dir / Path(save_file_name)
        else:
            logging.info("No save file path provided; data will not be saved to disk.")

    def append_embeddings(self, file_name: str, embeddings, dim = 768):
        emb_np = np.vstack(embeddings.values).astype("float32")
        n_new = emb_np.shape[0]
        
        if emb_np.shape[1] != dim:
            raise ValueError(f"Embeddings dimension mismatch: expected {dim}, got {emb_np.shape[1]}")
        
        file_path = self.data_dir / Path(file_name)
        if file_path.exists():
            #create file
            file_path.touch()
            mmap = np.memmap(file_path,
                             dtype="float32",
                             mode="w+",
                             shape=(n_new, dim))
            mmap[:] = emb_np
        else:
            old = np.memmap(file_path, 
                            dtype="float32",
                            mode="r+")
            n_old = old.size // dim
            
            mmap = np.memmap(file_path,
                             dtype="float32",
                             mode="w+",
                             shape=(n_old + n_new, dim))
            mmap[n_old:] = emb_np
            
        mmap.flush()
    
    def transform(self, data: pd.DataFrame):
        if self.save_file_path.exists() is True:
            data.to_csv(self.save_file_path, mode="a", index=False, header=False)
        else:
            data.to_csv(self.save_file_path, index=False, header=True)
