import zmq
import pickle5
import zlib
from api import *
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5554")

initial_setup()
while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))

    # query_name, args = message
    # response = functions[names.index(query_name)](*args)
    # compute something
    response = spatial_elimination(message)
    #  Send reply back to client
    print(type(response))
    socket.send(zlib.compress(pickle5.dumps(response)))