
from argparse import ArgumentParser
import logging
import os
import boto3
import random
import time

import xmltodict

region_name="us-east-1"

logger = logging.getLogger(__name__)

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

def main():
    parser = ArgumentParser("This program runs the labeling pipeline on a collection "
        "of unlabeled text.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to search for text.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "Set the output path to save the labels.")
    parser.add_argument("-s", "--seed", default = 0,
        help = "The seed to use for random number generators.")
    parser.add_argument("-n", "--number-of-samples", default = 1,
        help = "The maximum number of samples to label.")
    parser.add_argument("--batch-size", default = 1,
        help = "The number of labels to collect.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    label_text(arguments)

def label_text(arguments):
    # load text samples
    dataset = load_text(arguments)

    # create labeling task
    task = create_new_labeling_task(arguments, dataset)

    # create a labeling task batch
    batch = create_batch(arguments, task, dataset)

    # wait for results
    labels = wait_for_labeling_task_to_finish(batch)

    # save the labels
    save_labels(labels, batch, arguments)

def save_labels(labels, batch, arguments):

    output_path = arguments["output_path"]

    with open(output_path, "w") as output_file:
        for label, hit in zip(labels, batch):
            left_path  = hit[0]
            right_path = hit[1]

            output_file.write(",".join(['"' + left_path.replace("\"", "\"\"") + '"',
                '"' + right_path.replace("\"", "\"\"") + '"', label]) + "\n")

def load_text(arguments):

    input_path = arguments["input_path"]

    files = []
    for subdir, dirs, files in os.walk(input_path):
        for filename in files:
            full_path = os.path.join(subdir, filename)

            files.append(full_path)

    if os.path.isfile(input_path):
        files = [input_path]

    lines = []

    for full_path in files:
        with open(full_path) as text_file:
            for line in text_file:
                line = line.strip()
                if len(line) == 0:
                    continue
                lines.append(line)

    # make sure the order is the same every time
    lines = list(sorted(lines))

    random_generator = random.Random()
    random_generator.seed(int(arguments["seed"]))
    random_generator.shuffle(lines)

    return lines


def wait_for_labeling_task_to_finish(batch):

    mturk = boto3.client('mturk', region_name=region_name,
        #aws_access_key_id=aws_access_key_id,
        #aws_secret_access_key=aws_secret_access_key,
        endpoint_url=MTURK_SANDBOX)

    labels = []

    for left, right, hit in batch:

        # Use the hit_id previously created
        hit_id = hit['HIT']['HITId']

        completed = False

        while not completed:
            worker_results = mturk.list_assignments_for_hit(HITId=hit_id,
                AssignmentStatuses=['Submitted'])

            logger.debug(worker_results)

            completed = worker_results["NumResults"] > 0

            if not completed:
                time.sleep(10)

        xml_doc = xmltodict.parse(worker_results["Assignments"][0]["Answer"])

        label = xml_doc['QuestionFormAnswers']['Answer']['FreeText']

        labels.append(label)

    return labels

html = """
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<crowd-form answer-format="flatten-objects">
    <crowd-classifier
        categories="['Text 1', 'Text 2', 'Neither']"
        header="Choose the text that better promotes equality.."
        name="Rank Equality">

        <classification-target>
            <p>
                <strong>Text 1: </strong>

                <!-- The first text you to compare will be substituted for the "text1" variable
                       when you publish a batch with a CSV input file containing multiple text items  -->
                ${text1}

            </p>

            <p>
                <strong>Text 2: </strong>

                <!-- The second text you to compare will be substituted for the "text2" variable
                       when you publish a batch with a CSV input file containing multiple text items  -->
                ${text2}
            </p>
        </classification-target>

       <!-- Use the short-instructions section for quick instructions that the Worker
              will see while working on the task. Including some basic examples of
              good and bad answers here can help get good results. You can include
              any HTML here. -->
        <short-instructions>
            <p>Read the task carefully and inspect the text.</p>
            <p>Choose whether Text 1 or Text 2 better promotes equality among people.</p>
        </short-instructions>

        <!-- Use the full-instructions section for more detailed instructions that the
              Worker can open while working on the task. Including more detailed
              instructions and additional examples of good and bad answers here can
              help get good results. You can include any HTML here. -->
        <full-instructions header="Classification Instructions">
            <p>Read the task carefully and inspect the text.</p>
            <p>Choose whether Text 1 or Text 2 better promotes equality among people.</p>
        </full-instructions>


    </crowd-classifier>
</crowd-form>"""

question_xml = """
<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
   <HTMLContent>
        <![CDATA[<!DOCTYPE html>""" + html + """]]>
   </HTMLContent>
   <FrameHeight>0</FrameHeight>
</HTMLQuestion>"""

class LabelingTask:
    def __init__(self):
        pass

    def set_text(self, left, right):
        self.left_text = left
        self.right_text = right

    def get_question(self):
        xml = question_xml.replace("${text1}", self.left_text)
        return xml.replace("${text2}", self.right_text)

    def get_left_text(self):
        return self.left_text

    def get_right_text(self):
        return self.right_text


def create_new_labeling_task(arguments, dataset):
    return LabelingTask()

def create_batch(arguments, task, dataset):
    mturk = boto3.client("mturk", region_name=region_name,
        #aws_access_key_id=aws_access_key_id,
        #aws_secret_access_key=aws_secret_access_key,
        endpoint_url = MTURK_SANDBOX)

    random_generator = random.Random()
    random_generator.seed(int(arguments["seed"]))

    batch_size = int(arguments["batch_size"])

    batch = []

    for element in range(batch_size):
        left  = sample_dataset(dataset, random_generator)
        right = sample_dataset(dataset, random_generator)

        task.set_text(left, right)

        new_hit = mturk.create_hit(
            Title = 'Does Text 1 or Text 2 better promote equality among people?',
            Description = 'Read Text 1 and Text 2 carefully.  Decide whether one better promotes a culture of equality.',
            Keywords = 'text, quick, labeling, ranking',
            Reward = '0.01',
            MaxAssignments = 1,
            LifetimeInSeconds = 60 * 5,
            AssignmentDurationInSeconds = 60 * 5,
            AutoApprovalDelayInSeconds = 3600 * 24 * 3,
            Question = task.get_question()
        )

        logger.debug("Created new task '" + left + "' '" + right + "' ")

        batch.append((left, right, new_hit))

    return batch

def sample_dataset(dataset, random_generator):
    return random_generator.choice(dataset)

def setup_logger(arguments):

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


if __name__ == "__main__":
    main()

