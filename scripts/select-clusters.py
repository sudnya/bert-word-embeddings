
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def main():

    parser = ArgumentParser(description="A script for selecting specific set of clusters.")

    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-i", "--input-path", default="",
        help = "The path to the input cluster file.")
    parser.add_argument("-o", "--output-path", default="",
        help = "The path to the output cluster file.")

    arguments = vars(parser.parse_args())

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    selectClusters(arguments)

def selectClusters(arguments):
    inputPath = arguments["input_path"]
    outputPath = arguments["output_path"]

    clusters = {}
    clusterCounts = {}

    with open(inputPath) as histogramFile:
        while not isEndOfFile(histogramFile):
            if isCluster(histogramFile):
                cluster, count = parseCluster(histogramFile)
                clusterCounts[cluster] = count
                clusters[cluster] = {}
            else:
                word, count = parseWord(histogramFile)
                clusters[cluster][word] = count

    with open(outputPath, 'w') as histogramFile:
        for cluster, count in sorted(clusterCounts.items(), key=lambda x:x[1], reverse=True):
            words = clusters[cluster]

            if len(words) == 0:
                continue

            if excludeCluster(words):
                continue

            histogramFile.write("Cluster, " + str(cluster) + " (" + str(count) + ")\n")
            for word, wordCount in sorted(words.items(), key=lambda x:x[1], reverse=True):
                histogramFile.write("    '" + word + "' " + str(wordCount) + "\n")

excludedSet = set([' ', 'the', 'I', '/', 'to', ',', '_', ':', 'so', ')', "'", 'are',
    'from', 'and', 'or', 'not', 'it', '(', 'a', 'of', '.', 'this', 'have', 'on', 'an', '\n',
    'would', 'will', 'do', 'but', 'that', 'like', '[', 'as'])

def excludeCluster(words):
    mostFrequentWord = list(reversed(sorted(words.items(), key=lambda x: x[1])))[0][0]

    return mostFrequentWord in excludedSet

def isEndOfFile(openFile):
    char = openFile.read(1)

    openFile.seek(-1, 1)

    return len(char) == 0

def isCluster(openFile):
    nextLine = openFile.read(7)
    openFile.seek(-7, 1)

    return nextLine == "Cluster"

def parseCluster(openFile):
    line = openFile.readline()

    remainder = line[9:]

    logger.debug(str(("Remainder", remainder)))

    left, right = remainder.split(" ")

    return int(left), int(right.strip().strip("(").strip(")"))

def parseWord(openFile):
    line = readWordLine(openFile)
    logger.debug(str(("wordline", line)))

    wordStart = line.find("'")
    wordEnd = line.rfind("'")

    word = line[wordStart+1:wordEnd]
    count = int(line[wordEnd+1:].strip())

    return word, count

def readWordLine(openFile):
    inWord = False
    anyWordCharacters = False

    line = ""

    while True:
        char = openFile.read(1)

        if len(char) == 0:
            break

        line += char

        if char == "\n":
            if not inWord:
                break

        if char == "'":
            if inWord:
                if anyWordCharacters:
                    inWord = False
            else:
                inWord = True
                continue

        if inWord:
            anyWordCharacters = True

    return line



################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################





