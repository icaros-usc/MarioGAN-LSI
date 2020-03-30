# MarioGAN-LSI
An experimental setup for running quality diversity algorithms on GAN latent spaces.

# Instruction to run the experiment
At MarioGAN-LSI, run the command: python3 Search/run_search.py -w 1 -c Search/config/experiment/experiment.tml.
(-w is the workerID, ranges from 1 to total number of trials, -c is the path to the experiment config.)

# Config Format
One experiment config (in Search/config/experiment) points to multiple trials. One trial config (in Search/config/trial) points to an algorithm config and an elite map config (what features to use as BCs).

# Output
For each trial, 2 csv files are generated under /log/. One is the records and data gathered of all 10000 simulations, ordered by index 0~9999. The other is the elite map records, where each row contains all elements in the elite map at a given time step. The elite map is recorded after every 20 simulations, so there should be 10000/20 rows in total.
