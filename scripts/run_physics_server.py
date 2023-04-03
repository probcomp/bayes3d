import zmq
import pickle5
import zlib
import jax3dp3 as j

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5432")
physics_server = j.physics.PhysicsServer()

while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))
    response = physics_server.process_message(message)
    print(f"Sent response {response}...")
    socket.send(zlib.compress(pickle5.dumps(response)))