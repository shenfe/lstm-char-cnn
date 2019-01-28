#!/bin/bash

HDIM=650
WEDIM=128
CEDIM=15
HLAYERS=2
RLAYERS=2
MAXSTEP=10000000
MAXEPOCH=40
BATCH=20
DROP=0.5

python main.py \
	-mepoch $MAXEPOCH \
	-msteps $MAXSTEP \
	-hdim $HDIM \
	-wedim $WEDIM \
	-cedim $CEDIM \
	-hlayers $HLAYERS \
	-rlayers $RLAYERS \
	-dfreq 20 \
	-efreq 1000 \
	-maxlen 35 \
	-maxwdlen 65 \
	-batch $BATCH \
	-lr 20.0 \
	-drop $DROP \
	-clip 0.25 \
	-lr_decay 0.5 \
	# -cuda 
