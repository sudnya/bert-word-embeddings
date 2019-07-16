
import logging
from argparse import ArgumentParser
import boto3
import requests
import subprocess
import time
import sys

sys.path.append('source')

from serving.Vocab import Vocab

logger = logging.getLogger(__name__)

def main():

    parser = ArgumentParser(description="A script for deploying trained models to the server.")

    parser.add_argument("-s", "--server", default = "",
        help = "The name or ip of the server to deploy to.")
    parser.add_argument("-i", "--image-id", default = "ami-024a64a6685d05041",
        help = "Run the image id.")
    parser.add_argument("-k", "--key-pair", default = "macbook",
        help = "The key pair to use for the instance.")
    parser.add_argument("-m", "--model-path", default = "",
        help = "Deploy the specified model.")
    parser.add_argument("-q", "--query-text", default = "",
        help = "Query submit the specified text as a query to the existing server.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-V", "--vocab", default="/Users/gregorydiamos/checkout/lm/vocabs/vocab-guttenberg-256k.txt",
        help = "The path to the vocab file.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module")
    parser.add_argument("-O", "--override-config", default = [], action="append",
        help = "Override config file arguments")

    arguments = vars(parser.parse_args())

    setupLogger(arguments)

    if len(arguments["query_text"]) > 0:
        query(arguments)
    else:
        deploy(arguments)

def query(arguments):
    server = arguments["server"]

    isRunning, server = isServerRunning(server)

    if not isRunning:
        raise RuntimeError("Server '" + server + "' is not running...")

    url = "http://" + server + ":8501/v1/models/gender_equality_classifier:predict"

    logger.debug("Loading vocab: " + arguments["vocab"])
    vocab = Vocab(arguments["vocab"])
    queryText = arguments["query_text"]
    queryJson = {"inputs" : {"input_text" : formatQueryText(queryText, vocab)}}

    logger.debug("sending query text '" + queryText + "', json '" + str(queryJson) + "'")

    try:
        response = requests.post(url, json=queryJson)

        logger.debug("Response: " + str(response))
        logger.debug("Output: " + str(response.json()['outputs']['outputs'][0][0]))

        return response.status_code == 200
    except Exception as e:
        logger.debug(e)
        return False

def formatQueryText(text, vocab):

    tokens = tokenize(text, vocab)

    return [[[token, token] for token in tokens]]

def tokenize(text, vocab):
    characters = [character for character in text]

    tokens = []

    while True:
        token = tryMatchBestToken(characters, vocab)
        if token is None:
            break

        tokens.append(token)

    return tokens

def tryMatchBestToken(characters, vocab):
    if len(characters) == 0:
        return None

    possibleWord = characters[0]

    match = None

    for i in range(1, len(characters)):
        possibleWord += characters[i]

        if vocab.isPrefix(possibleWord):
            continue

        if not vocab.contains(possibleWord) and len(possibleWord) > 1:
            match = possibleWord[:-1]
            del characters[:i]
            break

    if match is None and vocab.contains(possibleWord):
        match = possibleWord
        del characters[:]

    if match is None:
        return None

    token = vocab.getToken(match)
    logger.debug("string: '" + match + "' -> " + str(token))

    return token


def deploy(arguments):
    server = arguments["server"]

    # Make sure that the server is running
    server = configureServer(server, arguments)

def configureServer(server, arguments):
    logger.debug("Checking if server '" + server + "' is running...")
    isRunning, server = isServerRunning(server)
    if isRunning:
        logger.debug("Checking if server '" + server + "' is running a model")
        if isServerRunningModel(server):
            return

        logger.debug(" starting up tensorflow serving ")
        runTensorflowServing(server, arguments)

        return

    logger.debug(" starting new server...")

    server = startServer(arguments)

def startServer(arguments):

    server = createInstance(arguments)

    runTensorflowServing(server, arguments)

def createInstance(arguments):

    ec2 = boto3.resource('ec2')

    instances = ec2.create_instances(
        ImageId=arguments["image_id"],
        MinCount=1,
        MaxCount=1,
        InstanceType="t2.micro",
        TagSpecifications=[{
            "ResourceType":"instance",
            "Tags":[{"Key":"GenderDefender", "Value":"0.1"}]
        }],
        SecurityGroupIds=[
            'tf-serving',
        ],
        KeyName=arguments["key_pair"])

    logger.debug(str(instances))

    client = boto3.client('ec2')

    while True:
        response = client.describe_instances(InstanceIds=[str(instance.id) for instance in instances])
        logger.debug(response)
        instanceDescription = response['Reservations'][0]['Instances'][0]
        ip = instanceDescription['PublicDnsName']
        if instanceDescription['State']['Name'] != 'pending':
            time.sleep(20)
            break

        time.sleep(5)

    return ip

def runTensorflowServing(server, arguments):
    copyModelToServer(server, arguments)

    runTensorflowServingOnRemoteServer(server)

def copyModelToServer(server, arguments):
    path = '/home/ubuntu/serving/tensorflow_serving/servables/random_gender_equality_classifier/1/'

    command = "ssh -o \"StrictHostKeyChecking no\" ubuntu@" + server + " mkdir -p " + path
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    logger.debug(command)

    command = ["rsync", '-avh',
        arguments["model_path"], "ubuntu@" + server + ":" + path]

    logger.debug(command)

    result = subprocess.run(command, stdout=subprocess.PIPE)

    logger.debug(result)

def runTensorflowServingOnRemoteServer(server):
    command = ["rsync", '-avh', 'source/ec2/setup-docker-tensorflow-serving.sh',
        "ubuntu@" + server + ":/home/ubuntu"]
    logger.debug(command)
    result = subprocess.run(command, stdout=subprocess.PIPE)
    logger.debug(result)

    command = ['ssh ubuntu@' + server + ' bash ' + 'setup-docker-tensorflow-serving.sh' ]

    logger.debug(command)

    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    logger.debug(result)

def isServerRunning(server):
    client = boto3.client('ec2')

    if len(server) > 0:
        response = client.describe_instances(Filters=[{"Name":"ip-address", "Values":[server]}])

        if len(response["Reservations"]) == 0:
            response = client.describe_instances(Filters=[{"Name":"dns-name", "Values":[server]}])
    else:
        response = client.describe_instances(Filters=[{"Name":"tag-key", "Values":["GenderDefender"]}])
        server = response['Reservations'][0]['Instances'][0]['PublicDnsName']

    logger.debug(str(response))

    isRunning = False

    if len(response["Reservations"]) != 0:
        isRunning = response['Reservations'][0]['Instances'][0]['State']['Name'] == 'running'

    return isRunning, server

def isServerRunningModel(server):
    url = "http://" + server + ":8501/v1/models/gender_equality_classifier"

    logger.debug("checking " + str(url))

    try:
        response = requests.get(url)

        logger.debug(str(response))

        return response.status_code == 200
    except Exception as e:
        logger.debug(e)
        return False

def setupLogger(arguments):

    if arguments["verbose"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################

