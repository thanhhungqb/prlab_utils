import json
from logging import Handler, LogRecord, getLogger, StreamHandler, makeLogRecord

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


class PrettyLineHandler(Handler):
    """
    Usage:
        logger.addHandler(PrettyLineHandler())
        or
        logger.addHandler(PrettyLineHandler(logging.FileHandler('/ws/tmp/t.txt', mode='w')))
    """
    def __init__(self, base_hdl=None, pretty_fn=None, **config):
        super(PrettyLineHandler, self).__init__(**config)

        self.level = 0
        self.filters = []
        self.lock = None

        # default is to standard output
        self.base = base_hdl if base_hdl is not None else StreamHandler()

        from tabulate import tabulate
        floatfmt = config.get('floatfmt', ".4f")
        self.pretty_fn = pretty_fn if pretty_fn is not None \
            else lambda js: str(tabulate([js], tablefmt="plain", floatfmt=floatfmt))

    def setLevel(self, level, **kwargs):
        self.level = level

    def addFilter(self, n_filter):
        self.filters.append(n_filter)

    def emit(self, record: LogRecord) -> None:
        try:
            st = self.pretty_fn(json.loads(record.msg))

            # clone record to not effect the chain of handlers
            n_record = makeLogRecord(record.__dict__)
            n_record.msg = st
            self.base.emit(n_record)

        except Exception as e:
            logger.warning(f"pretty line handler error {e}")
