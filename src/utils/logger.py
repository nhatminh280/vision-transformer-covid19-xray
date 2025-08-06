import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str,
    config: dict,
    log_dir: Path = Path("logs")
) -> logging.Logger:

    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    log_level = config.get("level", "INFO")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    log_format = config.get("format", 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.get("save_to_file", True):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / f"{datetime.now().strftime('%Y%m%d')}.log",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to set up file logging: {e}")
    
    return logger