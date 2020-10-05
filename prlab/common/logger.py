import json
from logging import Handler, LogRecord, getLogger

logger = getLogger(__name__)


class WandbHandler(Handler):
    """
    Use with logging, should be convert to json string (dumps) and log
    Usage:
        logger.addHandler(WandbHandler(name='test'))
        logger.info(json.dumps({'test': i}))
    """

    def __init__(self, **config):
        super(WandbHandler, self).__init__()

        import wandb
        self.wandb = wandb

        self.level = 0
        self.filters = []
        self.lock = None

        name = config.get('proj_name', 'default')
        wandb.init(config=config, project=name)

    def set_model(self, model):
        self.wandb.watch(model)
        return self

    def setLevel(self, level, **kwargs):
        self.level = level

    def addFilter(self, n_filter):
        self.filters.append(n_filter)

    def emit(self, record: LogRecord) -> None:
        # in case of Wandb, msg should be a dict of str:tensor to log
        try:
            self.wandb.log(json.loads(record.msg))
        except Exception as e:
            logger.warning(f"wandb log error {e}")
