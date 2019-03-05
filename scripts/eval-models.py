
import os

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

def main():
    models = getModels()
    datasets = getDatasets()

    results = []

    for dataset in datasets:
        for model in models:
            results.append(runEval(model, dataset))

    printResults(results)


main()


