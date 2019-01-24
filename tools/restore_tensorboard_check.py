import os

from maskrcnn_benchmark.utils.logger import TensorboardHandler


def norm_path(path, makedirs=False):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)

    if makedirs and not os.path.exists(path):
        os.makedirs(path)

    return path


class recorder():
    def __init__(self):
        self.msg = ''


def main(path):
    path = norm_path(path)
    log_file = os.path.join(path, 'log.txt')

    writer_tensor = TensorboardHandler(path)
    record = recorder()

    with open(log_file, 'rt') as flog:
        while True:
            line = flog.readline()
            if not line:
                break

            record.msg = line

            writer_tensor.emit(record)


if __name__ == '__main__':
    path = '~/lib/robin_cer/checkpoint/20190122-183720'
    main(path)