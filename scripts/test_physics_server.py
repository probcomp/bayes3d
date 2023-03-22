import os, time
from multiprocessing import Queue, Process
import numpy as np
import multiprocessing

def physics_main_loop(commands, results):
    import jax3dp3 as j
    physics_state = j.physics.PhysicsState()
    results.put("initial")

    while True:
        value = commands.get()
        if value is None:
            results.put(None)
            print("done")
            break

        command_type, args = value
        if command_type == "update":
            output = physics_state.update(args)
            results.put(output)
        elif command_type == "final_prediction":
            output = physics_state.final_prediction(args)
            results.put(output)

        time.sleep(0.05)

class PhysicsServer(object):
    def __init__(self):
        self.in_queue, self.out_queue = Queue(), Queue()
        self.worker = Process(target=physics_main_loop, args=(self.in_queue, self.out_queue))
        self.worker.start()
        x = self.out_queue.get()
        print(x)

    def update(self, rgb, depth):
        self.in_queue.put(("update", (rgb, depth,)))
        x = self.out_queue.get()
        return x
    
    def final_prediction(self):
        self.in_queue.put(("final_prediction", (None,)))
        x = self.out_queue.get()
        return x

server = PhysicsServer()

x = server.update(1,2)
print(x)


x = server.update(1,2)
print(x)


x = server.update(1,2)
print(x)


x = server.update(1,2)
print(x)


x = server.final_prediction()
print(x)


from IPython import embed; embed()
