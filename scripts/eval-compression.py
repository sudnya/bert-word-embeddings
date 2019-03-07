
import os
import subprocess

datasets = ["guttenberg-splits", "wikipedia"]
vocabDatasets = ["guttenberg", "wikipedia"]
vocabSizes = ["1k", "4k", "16k", "256k", "1m"]
dataSizes = [2**20//4, 2**24//4, 2**28//4]
unkModes = [("-unk", "--use-unk-tokenizer"), ("", "")]

workspace = "compression-experiments"
datasetRoot = "/data/1tb-ssd/language"
#datasetRoot = "/Users/gregorydiamos/checkout/bursty-lm/data"

def getLog():
    return open(os.path.join(workspace, "log"), "w+")

def approximatelyEqual(fileSize, dataSize):
    margin = 8

    if fileSize + 8 < dataSize:
        return False

    if dataSize + 8 < fileSize:
        return False

    return True


def runTokenizer(datasetPath, vocabPath, dataSize, unkMode, workspacePath):

    open(workspacePath, "a").close()

    command = ("time python train.py --make-test-set --test-set " + datasetPath +
        " --vocab-path " + vocabPath + " " + unkMode + " --test-set-size-bytes " + str(dataSize) +
        " -o " + workspacePath)

    print("Running '" + command + "'")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completedProcess = subprocess.run(command, env=env, shell=True, stdout=getLog(),
        stderr=subprocess.STDOUT)

    completedProcess.check_returncode()

    fileSize = os.path.getsize(workspacePath)

    if not approximatelyEqual(fileSize, dataSize):
        raise ValueError("Expected process to generate data with " + str(dataSize) +
            " bytes, but got file with " + str(fileSize) + " bytes")

    return fileSize

def runCompressor(workspacePath):
    command = ("time lzma " + workspacePath)

    print("Running '" + command + "'")

    env = os.environ.copy()
    completedProcess = subprocess.run(command, env=env, shell=True, stdout=getLog(),
        stderr=subprocess.STDOUT)

    completedProcess.check_returncode()

def countBytes(compressedPath):
    return os.path.getsize(compressedPath)

def runExperiment(dataset, vocabDataset, vocabSize, dataSize, unkMode):

    workspacePath = os.path.join(workspace, dataset + "-" + vocabDataset + "-vocab-" + vocabSize +
        "-datasize-" + str(dataSize) + unkMode[0])
    compressedPath = workspacePath + ".lzma"
    datasetPath = os.path.join(datasetRoot, "training", dataset)
    vocabPath = os.path.join(datasetRoot, "training", "vocabs", "vocab-" + vocabDataset + "-" + vocabSize + ".txt")

    fileSize = runTokenizer(datasetPath, vocabPath, dataSize, unkMode[1], workspacePath)
    runCompressor(workspacePath)

    compressedByteCount = countBytes(compressedPath)

    return (dataset, unkMode[0], vocabDataset, vocabSize, fileSize, compressedByteCount)

def saveResult(results):
    resultPath = os.path.join(workspace, "results.csv")

    with open(resultPath, "a") as resultFile:
        resultFile.write(", ".join([str(i) for i in results]) + "\n")


def clearWorkspace():
    import shutil

    if os.path.exists(workspace):
        shutil.rmtree(workspace)

    os.mkdir(workspace)

    resultPath = os.path.join(workspace, "results.csv")

    with open(resultPath, "w") as resultFile:
        resultFile.write("dataset, unk-mode, vocab-dataset, " +
            "vocab-size, file-size, compressed-size\n")

def main():
    clearWorkspace()

    for dataSize in dataSizes:
        for dataset in datasets:
            for vocabDataset in vocabDatasets:
                for unkMode in unkModes:
                    for vocabSize in vocabSizes:
                        saveResult(runExperiment(dataset, vocabDataset,
                            vocabSize, int(dataSize), unkMode))


main()






