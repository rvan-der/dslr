class DataDescriptionError(Exception):
    def __init__(self, message=""):
        super(DataDescriptionError, self).__init__()
        self.message = message

    def __str__(self):
        return self.message