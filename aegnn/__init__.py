import aegnn.asyncronous
import aegnn.utils
import aegnn.datasets
import aegnn.models

try:
    import aegnn.visualize
except ModuleNotFoundError:
    import logging
    logging.warning("AEGNN Module imported without visualization tools")

# Setup default values for environment variables, if they have not been defined already.
# Consequently, when another system is used, other than the default system, the env variable
# can simply be changed prior to importing the `aegnn` module.
aegnn.utils.io.setup_environment({
    # "AEGNN_DATA_DIR": "/remote-home/share/cjk/aegnn_new/aegnn/data/storage/",
    # "AEGNN_LOG_DIR": "/remote-home/share/cjk/aegnn_new/aegnn/data/scratch/"
    "AEGNN_DATA_DIR": "/home/cjk2002/code/event_code/aegnn_new/data/storage",
    "AEGNN_LOG_DIR": "/home/cjk2002/code/event_code/aegnn_new/data/scratch/"
})
