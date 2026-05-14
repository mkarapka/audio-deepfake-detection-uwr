import logging
from datetime import datetime

from src.common.constants import Constants as consts


def setup_logger(
    name: str = "audio_deepfake",
    level: int = logging.INFO,
    log_to_console: bool = False,
    log_to_file: bool = True,
) -> logging.Logger:
    """
    Konfiguruje i zwraca logger z zapisem do pliku w data/logs.

    Args:
        name: Nazwa loggera
        level: Poziom logowania (np. logging.DEBUG, logging.INFO)
        log_to_console: Czy wyświetlać logi w konsoli
        log_to_file: Czy zapisywać logi do pliku

    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format logów
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler do konsoli
    if log_to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Handler do pliku
    if log_to_file and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        # Twórz katalog logs jeśli nie istnieje
        consts.logs_dir.mkdir(parents=True, exist_ok=True)

        # Nazwa pliku z datą
        log_filename = f"{'audio_deepfake'}_{datetime.now().strftime('%Y-%m-%d')}.log"
        log_filepath = consts.logs_dir / log_filename

        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Domyślny logger dla projektu
def get_logger(name: str = "audio_deepfake") -> logging.Logger:
    """Pobiera istniejący logger lub tworzy nowy z domyślną konfiguracją."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def raise_error_logger(logger: logging.Logger, message: str, error_type: type = Exception):
    """Loguje błąd i podnosi wyjątek z podaną wiadomością."""
    logger.error(message)
    raise error_type(message)


class WandbLogger:
    def __init__(self, logger, run=None):
        self.run = run
        self.logger = logger

    def info(self, message: str):
        self.logger.info(message)
        if self.run is not None:
            self.run.log({"log": message})

    def log_metrics(self, metrics: dict, step: int = None):
        self.logger.info(f"{metrics} at step {step}")
        if self.run is not None:
            self.run.log(metrics, step=step)
