import logging

data_logger = logging.getLogger("preprocess")
train_logger = logging.getLogger("training")
eval_logger = logging.getLogger("evaluation")

def setup_logger(verbosity):
    """Configures the logger based on verbosity level."""
    levels = {
        0: logging.CRITICAL, # Silent, only critical errors
        1: logging.WARNING,  # Warnings and above
        2: logging.INFO,     # Info and above
        3: logging.DEBUG,    # Debug and above
    }
    format = "[%(name)s] [%(levelname)s] %(message)s"
    logging.basicConfig(level=levels.get(verbosity, logging.DEBUG), format=format)