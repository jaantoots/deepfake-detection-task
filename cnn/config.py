"""Classifier configuration"""

LOGGING = {
    "version": 1,
    "formatters": {
        "default": {"format": "%(asctime)s:%(levelname)s:%(message)s"},
        "file": {"format": "%(created)f:%(name)s:%(levelname)s:%(message)s"},
    },
    "handlers": {
        "stream": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": None,
            "formatter": "file",
            "level": "DEBUG",
        },
    },
    "root": {"handlers": ["stream", "file"]},
    "disable_existing_loggers": False,
}
DATA_ROOT = "../../data_train_wild_400"
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.0004
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
