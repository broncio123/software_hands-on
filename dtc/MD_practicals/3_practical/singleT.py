#!/usr/bin/env python

# load modules
import os
import sys
sys.path.append('cg_model/src')
import dynam_multithread as dyna
from datetime import datetime

start = datetime.now()

# make sure input more or less makes sense
# singleT(psfPath, coorPath, forcefieldPath, nsteps, nprint, temperature, outpath, threads, GPU)
usage = './singleT.py [psf] [cor] [forcefield.xml] [nsteps] [outpath]'
if len(sys.argv) != 6:
        print (usage)
        sys.exit()

# load psf file and check file extension
psf = sys.argv[1]
if psf.split('.')[-1] != 'psf':
	print (psf, 'does not appear to be a CHARMM protein structure file')
	sys.exit()

# load cor file and check file extension
cor = sys.argv[2]
if cor.split('.')[-1] != 'cor':
	print (cor, 'does not appear to be a CHARMM coordinate file')
	sys.exit()

# load xml file and check file extension
xml = sys.argv[3]
if xml.split('.')[-1] != 'xml':
	print (xml, 'does not appear to be an OpenMM format forcefield file')
	sys.exit()

# number of integration time steps to simulate
nsteps = int(sys.argv[4])

# label for output data
outpath = sys.argv[5]

# frequency with which data will be saved to file
nprint = 5000

# temperature for Langevin thermostat
temperature = 310

# used for parallelization - will not use in this tutorial, just run on 1 thread
threads = '1'

# if Boolean True, attempt to locate and run on a GPU
GPU = False

# use the singleT function from the dynam_multithread program to run the simulation
dyna.singleT(psf, cor, xml, nsteps, nprint, temperature, outpath, threads, GPU)

print ('Simulation complete; execution time:', datetime.now()-start)
