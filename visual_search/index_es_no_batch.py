import _init_paths

import cv2
import glob
import numpy as np
import argparse
import threadpool
import threading

from extractor import Extractor

from redis import Redis
from nearpy.storage import RedisStorage
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections, RandomBinaryProjectionTree, HashPermutations, HashPermutationMapper


import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Index image to elasticsearch')
    #parser.add_argument('--weight', dest='weight',
    #                    help='weight to test',
     #                   default=WEIGHT_PATH, type=str)
    #parser.add_argument('--model_path', dest='model_path',
     #                   help='path to the model',
      #                  default=MODEL_PATH, type=str)

    parser.add_argument('--input', dest='input',
                        help='Input image folder',
                        default=None, type=str)

    parser.add_argument('--es_host', dest='es_host',
                        help='es sever host',
                        default='localhost', type=str)
    parser.add_argument('--es_index', dest='es_index',
                        help='index name',
                        default='img_data', type=str)
    parser.add_argument('--es_type', dest='es_type',
                        help='index type',
                        default='obj', type=str)
    parser.add_argument('--es_port', dest='es_port',
                        help='es server port',
                        default=9200, type=int)

    args = parser.parse_args()

    #if not args.input:
     #   parser.error('Input folder not given')
    #return args

if __name__ == '__main__':
    args = parse_args()
    files = open("/home/images/wild_pics/picnames.txt")
    other = open("./no_faces.txt", "w")
    no_faces = []
    count = 0
    extractor = Extractor()

    dimension = 1024

    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 6)
    redis_storage = RedisStorage(Redis(host='10.211.55.6', port=6379, password="123456", db=0))
    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp], storage=redis_storage)
    redis_storage.store_hash_configuration(rbp)

    # Index 1000000 random vectors (set their data to a unique string)
    for line in files:
        count += 1
        im_path = line.strip()
        # create elasticsearch client
        # load images
        # images = glob.glob(args.input + "/*")

        # num_imgs = len(images)
        # read image
        im = cv2.imread(im_path)
        if im is None:
            no_faces.append(im_path)
        # im = im.astype(np.float32, copy=True)
        im_name = im_path.split('/')[-1]
        boxes = extractor.extract_regions_and_feats(im, im_name)
        for member in boxes.keys():
            print boxes[member]['f']
            print boxes[member]['cl']
            engine.store_vector(np.array(boxes[member]['f']), boxes[member]['cl'])

    files.close()
    for item in no_faces:
        other.write(item + "\n")
    other.close()
    # create feature extractor

