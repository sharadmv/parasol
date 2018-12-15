import os
from path import Path
import tensorflow as tf
import tempfile
import contextlib
import sys
from abc import ABCMeta, abstractmethod

gfile = tf.gfile

__all__ = ['tee_out']

class Tee:
    def __init__(self, original, target):
        self.original = original
        self.target = target

    def write(self, b):
        self.original.write(b)
        self.target.write(b)

@contextlib.contextmanager
def tee_out(out_dir):
    out_dir = Path(out_dir)
    stdout = tempfile.NamedTemporaryFile(delete=False)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stderr = tempfile.NamedTemporaryFile(delete=False)
    try:
        with StdoutTee(stdout.name, buff=1) as out, StderrTee(stderr.name, buff=1) as err:
            yield
    except:
        raise
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        with gfile.GFile(out_dir / 'stdout.log', 'w') as fp:
            with gfile.GFile(stdout.name, 'r') as out:
                fp.write(out.read())
        with gfile.GFile(out_dir / 'stderr.log', 'w') as fp:
            with gfile.GFile(stderr.name, 'r') as err:
                fp.write(err.read())
        os.remove(stdout.name)
        os.remove(stderr.name)


class Tee(object):
    """
    duplicates streams to a file.
    credits : http://stackoverflow.com/q/616645
    """

    def __init__(self, filename, mode="a", buff=0, file_filters=None, stream_filters=None):
        """
        writes both to stream and to file.
        file_filters is a list of callables that processes a string just before being written
        to the file.
        stream_filters is a list of callables that processes a string just before being written
        to the stream.
        both stream & filefilters must return a string or None.
        """
        self.filename = filename
        self.mode = mode
        self.buff = buff
        self.file_filters = file_filters or []
        self.stream_filters = stream_filters or []

        self.stream = None
        self.fp = None

    @abstractmethod
    def set_stream(self, stream):
        """
        assigns "stream" to some global variable e.g. sys.stdout
        """
        pass

    @abstractmethod
    def get_stream(self):
        """
        returns the original stream e.g. sys.stdout
        """
        pass

    def write(self, message):
        stream_message = message
        for f in self.stream_filters:
            stream_message = f(stream_message)
            if stream_message is None:
                break

        file_message = message
        for f in self.file_filters:
            file_message = f(file_message)
            if file_message is None:
                break

        if self.stream and stream_message is not None:
            self.stream.write(stream_message)

        if self.fp and file_message is not None:
            self.fp.write(file_message)

    def flush(self):
        if self.stream:
            self.stream.flush()
        if self.fp:
            self.fp.flush()
            os.fsync(self.fp.fileno())

    def __enter__(self):
        self.stream = self.get_stream()
        self.fp = open(self.filename, self.mode, self.buff)
        self.set_stream(self)

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.stream != None:
            self.set_stream(self.stream)
            self.stream = None

        if self.fp != None:
            self.fp.close()
            self.fp = None

    def isatty(self):
        return self.stream.isatty()

    def __repr__(self):
        return "<%s: %s>" % (self.__class__.__name__, self.filename)

    __str__ = __repr__
    __unicode__ = __repr__

class StdoutTee(Tee):
    def set_stream(self, stream):
        sys.stdout = stream

    def get_stream(self):
        return sys.stdout

class StderrTee(Tee):
    def set_stream(self, stream):
        sys.stderr = stream

    def get_stream(self):
        return sys.stderr
