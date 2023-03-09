#!/bin/bash

julia --project=. -e "using Gen; using ObjectLearning; 
simulate(ObjectLearning.model, ([(0, 0), (1, 1), (-1, -1)],))"
