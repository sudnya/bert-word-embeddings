###############################################################################
#
# \file    get-samples.py
# \author  Sudnya Diamos <sudnyadiamos@gmail.com>
# \date    Sunday June 9, 2019
# \brief   Given a path to dataset, get the specified number of samples that has
#          not already been labeled. Each sample contains body, src file, line
#          
###############################################################################

import argparse
import logging
import json
import random
import linecache

logger = logging.getLogger('get-samples')

def get_samples(input_file, number_of_samples):
    '''
    args: 
        input_file: path to the input dataset file
        number_of_samples: total output samples to be returned
    returns a list of json objects
    '''
    retVal = []
    total_samples_in_file = 0
    for line in open(input_file): 
        total_samples_in_file += 1
    
    logger.debug("Total samples in %s are %d", input_file,
                total_samples_in_file)

    random_indices = random.sample(range(1, total_samples_in_file), number_of_samples)
    logger.debug(random_indices)
    
    for i in random_indices:
        obj = {}
        obj['data'] = json.loads(linecache.getline(input_file, i)).get('body')
        obj['src']  = input_file
        obj['line_number'] = i
        
        logger.debug(obj)
        retVal.append(obj)
    return retVal


def main():
    parser = argparse.ArgumentParser(description="Get Samples from Dataset")
    parser.add_argument("-v", "--verbose", default = False, action = "store_true")
    parser.add_argument("-n", "--number_of_samples", default = 5)
    parser.add_argument("-i", "--input_file",
    default="/temp/datasets/reddit/reddit_sample1.json")

    parsedArguments    = parser.parse_args()
    arguments          = vars(parsedArguments)

    is_verbose         = arguments['verbose']
    number_of_samples  = arguments['number_of_samples']
    src_file           = arguments['input_file']

    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    output_samples = get_samples(src_file, number_of_samples)


   

if __name__ == '__main__':
    main()

