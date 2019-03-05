
from concurrent.futures import ThreadPoolExecutor
import humanize
import os
import subprocess

vocabSizes = ['1k', '4k', '16k', '64k', '256k', '1m']
dataSizes = [2**14, 2**17, 2**20, 2**24, 2**27]
configs = ['unk-trigram-all-data', 'trigram-all-data']
datasets = ['guttenberg', 'wikipedia']

processors = 12

def main():

    with ThreadPoolExecutor(max_workers=processors) as executor:
        futures = []

        for dataSize in dataSizes:
            for vocabSize in vocabSizes:
                for dataset in datasets:
                    for config in configs:
                        futures.append(executor.submit(runTraining,
                            vocabSize, dataSize, config, dataset))


        for future in futures:
            print("Finished '" + future.result() + "'")

def runTraining(vocabSize, dataSize, config, dataset):
    experimentName = (dataset + "-" + config + "-vocab-" + vocabSize + "-train-" +
        humanize.naturalsize(dataSize, gnu=True))

    command = ("time python train.py -n " + experimentName + " -c configs/" + config +
               ".json -O model.vocab=/data/1tb-ssd/language/training/vocabs/vocab-" + dataset +
               "-" + vocabSize + ".txt -O model.steps-per-epoch=" + str(dataSize) +
                " -O model.should-print-status=False" +
                " -L models.NgramModel")

    print("Launched: '" + command + "'")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completedProcess = subprocess.run(command, env=env, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    with open(os.path.join(experimentName + "-0", "stdout"), "wb") as stdout:
        stdout.write(completedProcess.stdout)

    return command

main()

