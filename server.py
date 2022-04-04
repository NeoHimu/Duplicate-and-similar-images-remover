from bottle import Bottle, run, request, response
import logging
logging.basicConfig(level=logging.INFO)
from json import dumps
from DuplicateRemover import DuplicateRemover
dirname = "images"

# Remove Duplicates
dr = DuplicateRemover(dirname)
# dr.find_duplicates()

# Find Similar Images
# dr.find_similar("images/6.webp",76)

logger = logging.getLogger("app.py")

app = Bottle()

#this is very important for other application to get access of the resource of the server
def allow_cors(func):
    """ this is a decorator which enable CORS for specified endpoint """
    def wrapper(*args, **kwargs):
        response.headers['Access-Control-Allow-Origin'] = '*' # * in case you want to be accessed via any website
        return func(*args, **kwargs)

    return wrapper

@app.route('/message')
def hello():
    return "Today is a beautiful day"  

@app.route("/urns", method='POST')
@allow_cors #this is very important
def creation_handler():
    '''Handles sentiment analysis for social media posts'''
    # return dumps({"message": "hello"})
    # parse input data
    try:
        urns = request.json.get('urns')
        # logger.info("request payload: "+str(urns))
    except:
        raise ValueError
    # clusters = dr.find_all_clusters(urns, 75)
    clusters = dr.ccd_vgg_all_clusters(urns)
    # return 200 Success
    response.headers['Content-Type'] = 'application/json'
    return dumps({'urns': clusters})


logger.info("server started")
run(app, host="127.0.0.1", port=8000)