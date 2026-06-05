from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.common.constants import Constants as consts
from src.common.logger import setup_logger
from src.preprocessing.audio_segmentator import AudioSegmentator
from src.preprocessing.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from src.preprocessing.feature_extractors.fft_extractor import FFTExtractor
from src.preprocessing.feature_extractors.wavlm_extractor import WavLmExtractor
from src.preprocessing.io.collector import Collector
from src.preprocessing.unique_audio_id_mapper import UniqueAudioIdMapper

UNKNOWN_SPEAKER_ID = -1


class InTheWildPreprocessingPipeline:
    def __init__(
        self,
        source_dir: Path = consts.in_the_wild_dir,
        meta_file: str = consts.in_the_wild_meta_file,
        files_chunk_size: int = 3_800,
    ):
        self.source_dir = Path(source_dir)
        self.audio_dir = self.source_dir / "audio"
        self.meta_path = self.source_dir / meta_file
        self.files_chunk_size = files_chunk_size

        self.label_by_stem: dict[str, str] = {}
        self.speaker_by_stem: dict[str, int] = {}

        self.logger = setup_logger(__class__.__name__, log_to_console=True)
        self.logger.info(f"Initialized InTheWildPreprocessingPipeline with source: {self.source_dir}")
        if not self.audio_dir.exists():
            self.logger.error(f"Audio directory does not exist: {self.audio_dir}")
        if not self.meta_path.exists():
            self.logger.error(f"Metadata file does not exist: {self.meta_path}")

    def _is_missing_columns(self, meta: pd.DataFrame) -> bool:
        missing_columns = {"file", "target"} - set(meta.columns)
        if missing_columns:
            self.logger.error(f"{self.meta_path} is missing required columns: {missing_columns}")
            return True
        return False

    def _list_audio_files(self) -> list[tuple[Path, str]]:
        meta = pd.read_csv(self.meta_path)
        if self._is_missing_columns(meta):
            return []

        speaker_map = dict(zip(meta["speaker"].unique(), pd.factorize(meta["speaker"])[0]))

        files: list[tuple[Path, str]] = []
        self.label_by_stem = {}
        self.speaker_by_stem = {}
        missing_files = 0
        for i, row in enumerate(meta.itertuples(index=False)):
            stem = Path(row.file).stem
            wav_path = self.audio_dir / row.file
            if not wav_path.exists():
                missing_files += 1
                continue
            self.label_by_stem[stem] = row.target
            self.speaker_by_stem[stem] = speaker_map.get(row.speaker, UNKNOWN_SPEAKER_ID)
            files.append((wav_path, stem))

        if missing_files:
            self.logger.warning(f"{missing_files} files listed in {self.meta_path.name} not found in {self.audio_dir}")
        self.logger.info(f"Prepared {len(files)} labelled files from {self.meta_path.name}")
        return files

    def _load_wav_record(self, wav_path: Path, stem: str) -> dict:
        array, sr = sf.read(str(wav_path))
        if array.ndim > 1:
            array = array.mean(axis=1)
        array = array.astype(np.float32)
        return {
            "__key__": stem,
            "wav": {"array": array, "sampling_rate": sr},
        }

    def _stream_records(self, files_chunk: list[tuple[Path, str]]):
        for wav_path, stem in files_chunk:
            yield self._load_wav_record(wav_path, stem)

    def _build_metadata(self, segs_metadata: pd.DataFrame) -> pd.DataFrame:
        record_ids = list(segs_metadata["key_id"])

        metadata = segs_metadata.copy()
        metadata["config"] = consts.in_the_wild
        metadata["split"] = consts.in_the_wild
        metadata["record_id"] = record_ids
        metadata["speaker_id"] = [self.speaker_by_stem[stem] for stem in record_ids]
        metadata["target"] = [self.label_by_stem[stem] for stem in record_ids]
        metadata = metadata.drop(columns=["key_id"])
        return metadata

    def _reset_output_files(self, collector: Collector):
        for file_path in (collector.get_metadata_file_path(), collector.get_embeddings_file_path()):
            if file_path.exists():
                self.logger.warning(f"Removing existing output file before fresh run: {file_path}")
                file_path.unlink()

    def _preprocess_dataset(self, file_name: str, feat_suffix: str, feature_extractor: BaseFeatureExtractor):
        if not file_name:
            self.logger.error("File name for saving processed data must be provided.")

        files = self._list_audio_files()
        if not files:
            self.logger.error(f"No WAV files found under {self.source_dir}; nothing to process.")
            return

        segmentator = AudioSegmentator(estimated_records_in_dataset=self.files_chunk_size)
        collector = Collector(save_file_name=file_name, feat_suffix=feat_suffix)
        uq_audio_id_mapper = UniqueAudioIdMapper()
        self._reset_output_files(collector)

        total_files = len(files)
        for start in range(0, total_files, self.files_chunk_size):
            chunk = files[start : start + self.files_chunk_size]
            chunk_end = start + len(chunk)
            self.logger.info(f"Processing files {start + 1}-{chunk_end} of {total_files}")

            segs_metadata, waves_segs = segmentator.transform(self._stream_records(chunk))
            if segs_metadata.empty:
                self.logger.warning("Chunk produced no segments, skipping.")
                continue
            self.logger.info(f"✓ Segmented into {segs_metadata.shape[0]} segments")

            modified_segs_metadata = self._build_metadata(segs_metadata)
            modified_segs_metadata = uq_audio_id_mapper.transform(metadata=modified_segs_metadata)
            self.logger.info(f"✓ Modified metadata ({len(modified_segs_metadata.columns)} columns)")

            embeddings = feature_extractor.transform(wave_segments=waves_segs)
            self.logger.info(f"✓ Extracted {len(embeddings)} embeddings")

            collector.transform(meta_df=modified_segs_metadata, embeddings=embeddings)
            self.logger.info(f"✓ Saved chunk to {file_name}\n")

    def preprocess_dataset_wavlm(self, file_name=consts.in_the_wild, batch_size=8):
        self._preprocess_dataset(
            file_name=file_name,
            feat_suffix=consts.wavlm_emb_suffix,
            feature_extractor=WavLmExtractor(batch_size=batch_size),
        )

    def preprocess_dataset_fft(self, file_name=consts.in_the_wild, batch_size=8):
        self._preprocess_dataset(
            file_name=file_name,
            feat_suffix=consts.fft_emb_suffix,
            feature_extractor=FFTExtractor(batch_size=batch_size),
        )
