
class sweep(object):

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values
