
class sweep(object):

    def __init__(self, values, names=None):
        self.values = values
        self.names = names

    def __iter__(self):
        if self.names is None:
            yield from zip(self.values, map(str, self.values))
        else:
            yield from zip(self.values, self.names)
