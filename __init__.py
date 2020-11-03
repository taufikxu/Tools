try:
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    # from . import cli
    from .cli.flags import FLAGS
    from .cli.config import init_cli, load_config

    from .logger.checkpointIO import CheckpointIO
    from .logger.metric_logger import Logger

    from . import evaluation
except ImportError as e:
    print(e)

init_cli()

if __name__ == "__main__":
    print(FLAGS)
    print(CheckpointIO)
    print(Logger)
