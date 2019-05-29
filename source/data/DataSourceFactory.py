
from data.TextDataSource import TextDataSource
from data.AOLQueryDataSource import AOLQueryDataSource

class DataSourceFactory:
    def __init__(self, config):
        self.config = config

    def create(self, sourceDescription):
        if sourceDescription["type"] == "TextDataSource":
            return TextDataSource(self.config, sourceDescription)

        if sourceDescription["type"] == "AOLQueryDataSource":
            return AOLQueryDataSource(self.config, sourceDescription)

        raise RuntimeError("Unknown data source type '" + self.config["type"] + "'")





