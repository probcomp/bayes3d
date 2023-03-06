import jax3dp3 as j
import zmq
import pickle5
import zlib

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5554")

intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=50.0
)
renderer = j.Renderer(intrinsics)


names = ["query1", "query2"]
functions = [query_type_1,query_type_2]

while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))
    print("Received request: {}".format(message))

    # query_name, args = message
    # response = functions[names.index(query_name)](*args)
    # compute something
    result = "0110011"

    #  Send reply back to client
    socket.send(zlib.compress(pickle5.dumps(result)))