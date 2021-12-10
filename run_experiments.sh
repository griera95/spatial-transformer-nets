#!/bin/bash

# run the three experiments

python run_eval_model.py
python run_eval_model.py model.coordconv=true
python run_eval_model.py model.stn.sampling=affine_diffeo