
import sys

sys.path.append('source')

from util.runLocally import runLocally

def addSlurmArgument(slurmArguments, arguments, name, defaultValue):
    value = defaultValue
    parameterName = name[2:]
    if parameterName in arguments:
        value = arguments[parameterName]
    slurmArguments.append(name + "=" + value)

def getSlurmArguments(arguments):
    slurmArguments = ["sbatch"]

    addSlurmArgument(slurmArguments, arguments, "--partition", "1080Ti")
    addSlurmArgument(slurmArguments, arguments, "--nodes", "1")
    addSlurmArgument(slurmArguments, arguments, "--ntasks", "1")
    addSlurmArgument(slurmArguments, arguments, "--gres", "gpu:1")
    addSlurmArgument(slurmArguments, arguments, "--cpus-per-task", "2")

    programArguments = []
    programArguments.append("python")
    programArguments.append("train.py")
    programArguments.extend(sys.argv[1:])
    programArguments.append("--no-use-slurm")

    slurmArguments.append("--wrap=\"" + " ".join(programArguments) + "\"")

    return slurmArguments

def runWithSlurm(arguments):
    import os
    import subprocess

    command = ' '.join(getSlurmArguments(arguments))

    print("Running slurm command: " + str(command))

    returnCode = subprocess.run(command, env=os.environ, shell=True).returncode

    if returnCode != 0:
        raise RuntimeError(str(command) + " -> " + str(returnCode))

def haveSlurm():
    import shutil
    srun = shutil.which('srun')

    if not srun is None:
        return True

    return False

def useSlurm(arguments):
    if not haveSlurm():
        return False

    return not arguments["no_use_slurm"]

def main():
    import logging
    from argparse import ArgumentParser

    parser = ArgumentParser(description="A script for launching a training process.")

    parser.add_argument("-n", "--experiment-name", default = "",
        help = "A unique name for the experiment.")
    parser.add_argument("-p", "--predict", default = False, action="store_true",
        help = "Run prediction on a specified trained model instead of training.")
    parser.add_argument("-m", "--model-path", default = "",
        help = "Load the specified model.")
    parser.add_argument("-c", "--config-file", default = "configs/sota.json",
        help = "The configuration parameters for the training program.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module")
    parser.add_argument("-O", "--override-config", default = [], action="append",
        help = "Override config file arguments")
    parser.add_argument("--no-use-slurm", default=False, action="store_true",
        help = "Disable the use of SLURM.  Note that if SLURM is not installed, "
               "it will not be used.")
    parser.add_argument("--make-test-set", default=False, action="store_true",
        help = "Create a test set from the validation set.")
    parser.add_argument("--test-set-size", default=20,
        help = "The number of graphs to sample from the validation set.")
    parser.add_argument("--test-set", default="",
        help = "The path to the test set to run on.")
    parser.add_argument("--vocab-size", default=10000,
        help = "The number of tokens to include in the vocab.")
    parser.add_argument("--make-vocab", default=False, action="store_true",
        help = "Make a vocab file for the validation set.")
    parser.add_argument("-o", "--output-directory", default="test-set",
        help = "The output directory the save the test set or vocab file.")

    arguments = vars(parser.parse_args())

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if useSlurm(arguments):
        runWithSlurm(arguments)
    else:
        runLocally(arguments)

################################################################################
## Guard Main
if __name__ == "__main__":
    main()
################################################################################

