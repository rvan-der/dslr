class DataError(Exception):
    def __init__(self, message=""):
        super(DataError, self).__init__()
        self.message = message

    def __str__(self):
        return self.message