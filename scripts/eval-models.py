
import os
import subprocess

modelDirectory = "models"
evalDatasetsDirectory = "/data/1tb-ssd/language/validation"


def getModels():
    models = []
    for prefix, directories, files in os.walk(modelDirectory):
        if 'config.json' in files:
            models.append(prefix)

    return models

def getDatasets():
    datasets = []

    for prefix, directories, files in os.walk(evalDatasetsDirectory):
        datasets += [os.path.join(prefix, f) for f in files]

    return datasets

def getPerplexity(outputString):
    #print("Getting perplexity from ", outputString)
    prefix = "Perplexity "
    start = outputString.find(prefix)
    if start < 0:
        print("Failed to parse output:\n", outputString)
        return None

    end = outputString.find("\n", start)

    try:
        perplexity = float(outputString[start+len(prefix):end])
        print("Perplexity " + str(perplexity))
        return perplexity
    except:
        print("Failed to parse output:\n", outputString)
        pass

    return None


def runEval(model, dataset):
    command = ("time python train.py -p -m " + model +
        " -O predictor.iterations=100 --test-set " + dataset +
        " -O model.discount-value=0.95")

    print("Running '" + command + "'")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completedProcess = subprocess.run(command, env=env, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    perplexity = getPerplexity(completedProcess.stdout.decode('utf-8'))

    return (model, dataset, perplexity)

def printResults(results):
    resultTable = {}

    models = set()
    datasets = set()

    for model, dataset, perplexity in results:
        models.add(model)
        datasets.add(dataset)

        resultTable[(model, dataset)] = perplexity

    print("experiment-name, " + ", ".join(sorted(datasets)))

    for model in sorted(models):
        print(model + ", " + ", ".join([str(resultTable[model, dataset]) for dataset in sorted(datasets)]))

def main():
    models = getModels()
    datasets = getDatasets()

    results = []

    for dataset in sorted(datasets):
        for model in sorted(models):
            results.append(runEval(model, dataset))

    printResults(results)

main()


