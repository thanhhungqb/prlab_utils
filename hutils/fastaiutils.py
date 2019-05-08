from fastai import callbacks
from fastai.callbacks import SaveModelCallback, partial


def get_callbacks(best_name='best', monitor='accuracy', csv_filename='log', csv_append=True):
    out = [
        partial(SaveModelCallback, monitor=monitor, name=best_name),
        callbacks.TrackerCallback,
        partial(callbacks.CSVLogger, filename=csv_filename, append=csv_append)
    ]
    return out
