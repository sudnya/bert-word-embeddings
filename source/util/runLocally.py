import sys
import inspect
import json
import os
import logging
import shutil

from models.ModelFactory import ModelFactory
from inference.Predictor import Predictor
from data.DataSources import DataSources
from data.DataSourceFactory import DataSourceFactory
from mpi.mpi import mpi
from util.vocab import saveVocab

import tensorflow as tf
from tensorflow.python.client import device_lib

def getAvailableGpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def hasGpus():
    return len(getAvailableGpus()) > 0

def getDevice():
    if hasGpus():
        return "/device:GPU:0"
    else:
        return "/cpu:0"

def getData(sources, config):
    dataSources = DataSources(config)

    for source in sources:
        dataSources.addSource(DataSourceFactory(config).create(source))

    return dataSources

def getTrainingData(config):
    return getData(config["trainingDataSources"], config)

def getValidationData(config):
    return getData(config["validationDataSources"], config)

def getModel(config, trainingData, validationData):
    return ModelFactory(config, modelName=config["model"]["type"],
        trainingData=trainingData, validationData=validationData).create()

def getPredictor(config, validationData):
    return Predictor(config, validationData)

def saveData(validationData, tokenCount, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    generator = validationData.getGraphGenerator()

    with open(os.path.join(directory, "data.txt")) as outputFile:
        for i in range(tokenCount):
            token = generator.getNextToken()
            outputFile.write(token.getString())

def nameDirectory(directory):
    extension = 0

    directory = os.path.abspath(directory)

    while os.path.exists(directory + '-' + str(extension)):
        extension += 1

    return directory + '-' + str(extension)

def makeExperiment(config):
    if mpi.isMasterRank():
        directory = config["model"]["directory"]

        if not os.path.exists(directory):
            os.makedirs(directory)

        # save the config file
        configPath = os.path.join(directory, "config.json")

        with open(configPath, 'w') as outfile:
            json.dump(config, outfile, indent=4, sort_keys=True)

        # save the code
        codeDirectory = os.path.join(directory, "source")
        codeFile = os.path.join(directory, "train.py")

        currentFilePath = os.path.abspath(inspect.getfile(inspect.currentframe()))
        currentCodePath = os.path.dirname(os.path.dirname(currentFilePath))
        trainingFilePath = os.path.join(os.path.dirname(currentCodePath), "train.py")

        shutil.copytree(currentCodePath, codeDirectory, ignore=shutil.ignore_patterns(("^.py")))
        shutil.copyfile(trainingFilePath, codeFile)

    mpi.barrier()

def loadConfig(arguments):
    if arguments["model_path"] != "":
        arguments["config_file"] = os.path.join(arguments["model_path"], 'config.json')

    try:
        with open(arguments["config_file"]) as configFile:
            config = json.load(configFile)
    except:
        config = {}

    if len(arguments["test_set"]) > 0:
        config["validationDataSources"] = [{ "type" : "TextDataSource",
                                             "path" : arguments["test_set"] }]


    return config

def overrideConfig(config, arguments):
    for override in arguments["override_config"]:
        path, value = override.split('=')
        components = path.split('.')

        localConfig = config
        for i, component in enumerate(components):
            if i == len(components) - 1:
                localConfig[component] = value
            else:
                if not component in localConfig:
                    localConfig[component] = {}
                localConfig = localConfig[component]

def runLocally(arguments):
    import numpy

    numpy.set_printoptions(precision=3, linewidth=150)

    device = getDevice()
    with tf.device(device):
        for scope in arguments["enable_logger"]:
            logger = logging.getLogger(scope)
            logger.setLevel(logging.DEBUG)

        config = loadConfig(arguments)

        overrideConfig(config, arguments)

        if arguments["predict"]:

            if not "predictor" in config:
                config["predictor"] = {}

            if "model" in config:
                config["model"]["directory"] = arguments["model_path"]

            validationData = getValidationData(config)
            predictor = getPredictor(config, validationData)
            perplexity = predictor.predict()

            print("Perplexity " + str(perplexity))

        elif arguments["make_test_set"]:
            validationData = getValidationData(config)

            saveData(validationData, int(arguments["test_set_size"]),
                arguments["output_directory"])

        elif arguments["make_vocab"]:
            validationData = getValidationData(config)

            saveVocab(validationData, int(arguments["vocab_size"]),
                arguments["output_directory"])

        else:
            config["model"]["directory"] = nameDirectory(arguments["experiment_name"])

            makeExperiment(config)

            trainingData = getTrainingData(config)
            validationData = getValidationData(config)

            model = getModel(config, trainingData, validationData)
            model.train()


