

from argparse import ArgumentParser
import logging
import csv
import elo

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts ranked text to ELO stored in a CSV.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to a csv of ranked text.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")
    parser.add_argument("-L", "--enable-logger", default = [], action="append",
        help = "Enable logging for a specific module.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "Set the output path to save the labels.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_labels_to_elo(arguments)

def convert_labels_to_elo(arguments):
    current_elo = {}
    elo_environment = elo.Elo()

    with open(arguments['input_path'], "r") as csvfile:
        for row in csv.reader(csvfile, delimiter=',', quotechar='"'):
            if row[2].strip() == 'Text 1':
                add_elo_win(elo_environment, current_elo, row[0], row[1])
            elif row[2].strip() == 'Text 2':
                add_elo_win(elo_environment, current_elo, row[1], row[0])
            else:
                add_elo_win(elo_environment, current_elo, row[1], row[0], draw=True)


    with open(arguments['output_path'], 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')

        for text, score in sorted(current_elo.items(), key=lambda x: x[1]):
            writer.writerow([text, score])


def add_elo_win(elo_environment, scores, winner, loser, *, draw=False):

    if not winner in scores:
        scores[winner] = elo_environment.create_rating()

    if not loser in scores:
        scores[loser] = elo_environment.create_rating()

    scores[winner], scores[loser] = elo_environment.rate_1vs1(
        scores[winner], scores[loser], drawn=draw)

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

