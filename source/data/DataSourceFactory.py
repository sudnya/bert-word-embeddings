
from data.TextDataSource import TextDataSource
from data.AOLQueryDataSource import AOLQueryDataSource
from data.RedditDataSource import RedditDataSource
from data.RankingCsvDataSource import RankingCsvDataSource

class DataSourceFactory:
    def __init__(self, config):
        self.config = config

    def create(self, sourceDescription):
        if sourceDescription["type"] == "TextDataSource":
            return TextDataSource(self.config, sourceDescription)

        if sourceDescription["type"] == "AOLQueryDataSource":
            return AOLQueryDataSource(self.config, sourceDescription)

        if sourceDescription["type"] == "RedditDataSource":
            return RedditDataSource(self.config, sourceDescription)

        if sourceDescription["type"] == "RankingCsvDataSource":
            return RankingCsvDataSource(self.config, sourceDescription)

        raise RuntimeError("Unknown data source type '" + sourceDescription["type"] + "'")





