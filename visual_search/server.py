#!/usr/bin/env python


from flask import Flask
from flask import request, jsonify, \
    send_from_directory

from base64 import b64encode
from elasticsearch import Elasticsearch
import numpy as np
from nearpy import Engine
from nearpy.storage import RedisStorage
from nearpy.hashes import RandomBinaryProjections
from redis import Redis
import time
from nearpy.filters import NearestFilter, UniqueFilter
from nearpy.distances import CosineDistance,EuclideanDistance

from extractor import Extractor
from lib.utils.im_util import read_img_blob
from lib.es.ImFea_pb2 import ImFea, ImFeaArr, \
    ImFeaBinArr, ImFeaBin

IMGS_PATH = '/home/images/wild_pics/'
app = Flask(__name__, static_url_path='')


class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def load_model():
    """Load feature extractor model"""
    extractor = Extractor()
    return extractor


extractor = load_model()
es = Elasticsearch(hosts='localhost:9200')


@app.route("/hello", methods=['GET'])
def hello():
    return "Hello, world!"


@app.route("/extract_fea", methods=['GET', 'POST'])
def extract_fea():
    imgStr = request.values.get('img')
    if imgStr is None:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        img = read_img_blob(imgStr)
    except:
        raise InvalidUsage('Invalid "img" param, must be a base64 string',
                           status_code=410)
    fea = extractor.extract_imfea(img)
    is_binary = request.values.get('is_binary')
    if is_binary and is_binary == 'true':
        fea = extractor.binarize_fea(fea)
        fea_obj = ImFeaBin()
    else:
        fea_obj = ImFea()
    fea_obj.f.extend(fea)
    base64str = base64.b64encode(fea_obj.SerializeToString())

    out = {}
    out['fea'] = base64str
    return jsonify(out)


@app.route("/get_tags", methods=['GET', 'POST'])
def get_tags():
    """get tags corresponding to a image"""
    if not 'img' in request.files:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        f = request.files.get('img')
        img_str = f.read()
        img = read_img_blob(img_str)
    except:
        raise InvalidUsage('Invalid "img" param, must be a blob string',
                           status_code=410)
    tags = extractor.get_tags(img, img_str)
    out = {}
    out['tags'] = tags
    return jsonify(out)


QUERY = """
{
	"sort": [{"_score": "desc"}],
	"fields": ["name"],
	"query": {
		"image": {
			"image": {
				"image": "##fea##",
				"feature": "CEDD",
				"hash": "LSH"
			}
		}
	}
}
"""


@app.route("/search", methods=['GET', 'POST'])
def search():
    """get tags corresponding to a image"""
    redis_object = Redis(host='10.211.55.6', port=6379, password="123456", db=0)
    redis_storage = RedisStorage(redis_object)

    # Get hash config from redis
    config = redis_storage.load_hash_configuration('rbp')
    dimension = 1024
    if config is None:
        # Config is not existing, create hash from scratch, with 10 projections
        print "Noet exit"
        exit()
        # lshash = RandomBinaryProjections('rpb', 10)
    else:
        # Config is existing, create hash with None parameters
        lshash = RandomBinaryProjections(None, None)
        # Apply configuration loaded from redis
        lshash.apply_config(config)
    engine = Engine(dimension, lshashes=[lshash], distance=EuclideanDistance(), vector_filters=[NearestFilter(10)],
                    storage=redis_storage)
    results = []
    if not 'img' in request.files:
        raise InvalidUsage('parameter "img" is missing', status_code=410)
    try:
        f = request.files.get('img')
        img_str = f.read()
        img = read_img_blob(img_str)
    except:
        raise InvalidUsage('Invalid "img" param, must be a blob string',
                           status_code=410)
    fea = extractor.extract_imfea(img)
    if fea is None:
        print "No faces detected"
        return jsonify({})
    # fea = extractor.binarize_fea(fea)
    # fea_str = ','.join([str(int(t)) for t in fea])
    for fea_i in fea:
        results.append(engine.neighbours(np.array(fea_i)))

    rs = []
    for result_i in results:
        for result_j in result_i:
            o = {}
            o['score'] = result_j[2]
            # distinct
            all_imgs = set([])
            o['im_src'] = result_j[1]
            im_src = '/img/{}'.format(o['im_src'])
            if not im_src in all_imgs:
                o['im_src'] = im_src
                all_imgs.add(im_src)
                rs.append(o)
    print rs
    out = {}
    out['hits'] = rs
    return jsonify(out)


@app.route('/static/<path:path>')
def send_static_files(path):
    "static files"
    return send_from_directory('static_data', path)


@app.route('/img/<path:path>')
def send_image(path):
    "static files"
    return send_from_directory(IMGS_PATH, path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
