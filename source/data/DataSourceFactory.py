
from data.TextDataSource import TextDataSource

class DataSourceFactory:
    def __init__(self, config):
        self.config = config

    def create(self, sourceDescription):
        if sourceDescription["type"] == "TextDataSource":
            return TextDataSource(self.config, sourceDescription)

        raise RuntimeError("Unknown data source type '" + self.config["type"] + "'")





