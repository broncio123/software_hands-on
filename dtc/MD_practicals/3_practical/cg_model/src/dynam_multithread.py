#!/usr/bin/env python

"""

### Version and authorship

Original version: 2-Feb-2020
Current version: 1-Apr-2020

Daniel A. Nissley, Ph.D.
Oxford Protein Informatics Group
Department of Statistics
University of Oxford

### Purpose

Run molecular dynamics simulation as required mandated by replica_exchange_manager.py

### Notes

(1) One call to this script needs to be made for each replica after each exchange attempt during replica exchange.
(2) This version implements velocity rescaling.
(3) Consider consolidating main() to be more concise - seems a bit diffuse at the moment
(4) Had to move the block of code in addNativeContacts to main() to avoid messy issues with how it modifies the system object
    If I was a better programmer (or cared to try) I could probably still use the function if I mess around with pointers
(5) Avoid the use of sys.exit() in this script, it will cause replica_exchange_manager.py to hang because it submits calls
    to this script using multiprocessing. Not sure why, exactly.
(6) Added ability to run temperature quenching simulations via the tempQuench function
(7) Added ability to run single-temperature simulations using the singleT function
(8) Added ability to use GPUs as requested in the rex cntrl file or in the singleT command - have not yet updated tempQuench to use GPUs
(9) Updated to use new .xml file format that contains all forcefield information and does not require the addition of a custom bond force
    to simulate the influence of native contacts
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import sys
from datetime import datetime
import numpy as np

### functions
def main(psfPath, coorPath, forcefieldPath, nsteps, replica, temperature, restart, exchange, outpath, velocityScale, threads, GPU):

	# psfPath: path to protein structure file for this run
	# coorPath: path to coordinate file for this run - only used for first iteration, then coordinates are loaded from the checkpoint file
	# forcefieldPath: path to OpenMM forcefield.xml file
	# nsteps: integer number of integration time steps to be simulated
	# replica: the integer designation of this replica (1, ..., N)
	# temperature: the temperature in Kelvin at which this dynamics run will be simulated
	# restart: True or False; if True, restart from a checkpoint file
	# exchange: integer number of this exchange attempt
	# outpath: output directory for simulation data
	# velocityScale: floating point by which all velocities will be collectively scaled before performing the dynamics run
	# threads: number of threads to use for this dynamics run - must be a string that can be converted to an integer
	# GPU: Boolean; if True, attempt to find and run on a GPU using the CUDA platform

	### set up and run dynamics

	# get input files and key parameters
	friction = 0.05                         # friction coefficient in inverse picoseconds
	tstep = 15                              # integration time step in femtoseconds
	psf = CharmmPsfFile(psfPath)            # load CHARMM-format Protein Structure File from which to extract topology
	coor = CharmmCrdFile(coorPath)          # load CHARMM-format coordinate file from which to extract initial coordinates
	forcefield = ForceField(forcefieldPath) # load OpenMM-format forcefield.xml file from which to extract forcefield parameters

	# generate residue template dictionary - this is required for the CA CG model because OpenMM will attempt to guess
	# residue templates but fail due to the non-standard labelling scheme, leading to errors
	template_map = {}
	for chain in psf.topology.chains():
	        for res in chain.residues():
        	        template_map[res] = res.name

	# generate system object and set up specifics for the non-bonded force
	system = forcefield.createSystem(psf.topology, residueTemplates=template_map, constraints=AllBonds,
                                         nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0*nanometer,
                                         switchDistance=1.8*nanometer, removeCMMotion=True)
	nbForce = system.getForce(3)
	nbForce.setUseSwitchingFunction(True)
	nbForce.setSwitchingDistance(1.8*nanometer)

	# set up integrator
	integrator = LangevinIntegrator(temperature*kelvin, friction/picosecond, tstep*femtoseconds)
	integrator.setConstraintTolerance(0.00001)

	# determine which platform to use
	if GPU == True:
		# then try to use CUDA
		platform = Platform.getPlatformByName('CUDA')
		# set up the Simulation object
		simulation = Simulation(psf.topology, system, integrator, platform, {'CudaPrecision': 'mixed'})
	else:
		# use CPUs
		platform = Platform.getPlatformByName('CPU')
		# set up the Simulation object
		simulation = Simulation(psf.topology, system, integrator, platform, {'Threads': threads})

	# set up a reporter to print the output data
	simulation.reporters.append(DCDReporter(outpath+'replica'+replica+'/p'+exchange+'.dcd', nsteps))

	# set starting coordinates
	if restart: # if we are restarting this run
		# load the checkpoint file to restart exactly where we left off; need to check harder in the future if this is the correct way to do this - will this use the old temperature?
		#simulation.context.loadCheckpoint('replica'+str(replica)+'/checkpoint.chk')
		with open (outpath+'replica'+replica+'/checkpoint'+str(int(exchange)-1)+'.chk', 'rb') as f: # try reading it the same way as in the example online
			simulation.context.loadCheckpoint(f.read())
	else: # then this must be the intial run, so just use the starting coordinates
		simulation.context.setPositions(coor.positions)

	# if velocityScale == 'False' then we will just use the velocities as stored in the checkpoint file, which should continue the simulation seemlessly
	# if velocityScale is a float instead...
	if velocityScale != 'False':
		velocityScale = np.float64(velocityScale)
		# get the current velocities after loading the checkpoint file
		temp = simulation.context.getState(getVelocities=True)
		veloc = temp.getVelocities(asNumpy=True) # get the velocities as an array
		veloc = velocityScale*veloc # scale the velocities as necessary
		simulation.context.setVelocities(veloc) # set the velocities

	# run dynamics
	simulation.step(nsteps)

	### save the simulation checkpoint file

	# write out the potential energy of the final coordinates
	eneLog = open(outpath+'replica'+replica+'/ene.log', 'a')
	eneLog.write(str(simulation.context.getState(getEnergy=True).getPotentialEnergy())+'\n')
	eneLog.close()

	# write out the temperature at which this replica was run
	tempLog = open(outpath+'replica'+replica+'/temp.log', 'a')
	tempLog.write(str((temperature))+'\n')
	tempLog.close()

	# save the checkpoint file so this run can be restarted at the next iteration
	simulation.saveCheckpoint(outpath+'replica'+replica+'/checkpoint'+str(exchange)+'.chk')

	return

# function to perform a single-temperature simulation without doing all of the replica-exchange specific things
def singleT(psfPath, coorPath, forcefieldPath, nsteps, nprint, temperature, outpath, threads, GPU):

	# psfPath: path to protein structure file for this run as a string
	# coorPath: path to coordinate file for this run as a string
	# forcefieldPath: path to OpenMM forcefield.xml file as a string
	# nsteps: integer number of integration time steps to be simulated as an integer
	# nprint: number of steps to simulate between saving data to DCD file as an integer
	# temperature: the temperature in Kelvin at which this dynamics run will be simulated as a float
	# outpath: output directory/name for simulation data - the output DCD file will be saved as outpath+'.dcd', which is different from the
	#          way outpath is used in the replica exchange script (taken as a string)
	# threads: number of threads to use for this simulation as a string
	# GPU: Boolean; if True, use CUDA on GPUs; otherwise, use CPUs

	### set up and run dynamics

	start = datetime.now()

	# get input files and key parameters
	friction = 0.05                         # friction coefficient in inverse picoseconds
	#friction = 0.20                         # friction coefficient in inverse picoseconds
	tstep = 15                              # integration time step in femtoseconds
	psf = CharmmPsfFile(psfPath)            # load CHARMM-format Protein Structure File from which to extract topology
	coor = CharmmCrdFile(coorPath)          # load CHARMM-format coordinate file from which to extract initial coordinates
	forcefield = ForceField(forcefieldPath) # load OpenMM-format forcefield.xml file from which to extract forcefield parameters

	# generate residue template dictionary - this is required for the CA CG model because OpenMM will attempt to guess
	# residue templates but fail due to the non-standard labelling scheme, leading to errors
	template_map = {}
	for chain in psf.topology.chains():
	        for res in chain.residues():
        	        template_map[res] = res.name

	# generate system object and set up specifics for the non-bonded force
	system = forcefield.createSystem(psf.topology, residueTemplates=template_map, constraints=AllBonds,
                                         nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0*nanometer,
                                         switchDistance=1.8*nanometer, removeCMMotion=True)
	nbForce = system.getForce(3)
	nbForce.setUseSwitchingFunction(True)
	nbForce.setSwitchingDistance(1.8*nanometer)

	# set up integrator
	integrator = LangevinIntegrator(temperature*kelvin, friction/picosecond, tstep*femtoseconds)
	integrator.setConstraintTolerance(0.00001)

	# determine which platform to use for this simulation
	if GPU == True: # use GPUs with CUDA (NVIDIA only?)
		platform = Platform.getPlatformByName('CUDA')
		# set up the Simulation object
		simulation = Simulation(psf.topology, system, integrator, platform, {'CudaPrecision': 'mixed'})
	else: # use CPUs
		platform = Platform.getPlatformByName('CPU')
		# set up the Simulation object
		simulation = Simulation(psf.topology, system, integrator, platform, {'Threads': threads})

	# set up reporters for printing the output data
	simulation.reporters.append(DCDReporter(outpath+'.dcd', nprint))
	simulation.reporters.append(StateDataReporter(outpath+'.log', nprint, step=True, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, separator='\t'))

	# set the initial coordinates
	simulation.context.setPositions(coor.positions)

	# profile the energies
	#print ('BEFORE MINIMIZATION')
	#profileEnergies(system, simulation)
	#simulation.minimizeEnergy()

	#print ('AFTER MINIMIZATION')
	#profileEnergies(system, simulation)

	# run dynamics
	simulation.step(nsteps)

	#print ('AFTER DYNAMICS')
	#profileEnergies(system, simulation)

	#print ('Run time:', datetime.now()-start)

	return

# function to convert the string 'True' into Boolean True and all other inputs to Boolean False
def str2bool(s):
	if s == 'True':
		return True
	else:
		return False

# protocol for temperature quenching simulations
def tempQuench(psfPath, coorPath, forcefieldPath, nstepsEquil, nstepsQuench, nprint, equilTemp, quenchTemp, outpath, threads, friction, GPU, restart, resNum):

	# psfPath: string, path to protein structure file containing the topology for this run
	# coorPath: string, path to initial coordinates file for this run
	# forcefieldPath: string, path to OpenMM format .xml forcefield file for this run
	# nstepsEquil: integer, number of integration time steps to simulate at equilTemp
	# nstepsQuench: integer, number of integration time steps to simulate at quenchTemp
	# nprint: integer, number of integration time steps between saving frames to file
	# equilTemp: float, high temperature to force unfolding
	# quenchTemp: float, low temperature at which to quench to the folded state
	# outpath: string, output label for data; will be used for both dcd and log files
	# threads: string, number of threads on which to run the simulation
	# friction: float, the collision frequency in inverse picoseconds for this run
	# GPU: Boolean; if true, attempt to run using the CUDA platform
	# restart: Boolean, determines whether or not we attempt to restart from the checkpoint file written at the end of a previous run
	# resNum: integer number of this restart

	# returns: nothing, but writes trajectory coordinates to outpath+'.dcd' and energies and temperature information to outpath+'.log'

	### set up and run dynamics for initial high-temperature equilibration

	# get input files and key parameters
	#friction = 0.05                         # friction coefficient in inverse picoseconds
	tstep = 15                              # integration time step in femtoseconds
	psf = CharmmPsfFile(psfPath)            # load CHARMM-format Protein Structure File from which to extract topology
	coor = CharmmCrdFile(coorPath)          # load CHARMM-format coordinate file from which to extract initial coordinates
	forcefield = ForceField(forcefieldPath) # load OpenMM-format forcefield.xml file from which to extract forcefield parameters

	# generate residue template dictionary - this is required for the CA CG model because OpenMM will attempt to guess
	# residue templates but fail due to the non-standard labelling scheme, leading to errors
	template_map = {}
	for chain in psf.topology.chains():
	        for res in chain.residues():
        	        template_map[res] = res.name

	# generate system object and set up specifics for the non-bonded force
	system = forcefield.createSystem(psf.topology, residueTemplates=template_map, constraints=AllBonds,
                                         nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=2.0*nanometer,
                                         switchDistance=1.8*nanometer, removeCMMotion=True)
	nbForce = system.getForce(3)
	nbForce.setUseSwitchingFunction(True)
	nbForce.setSwitchingDistance(1.8*nanometer)

	if restart == False:

		resNum = 0
		# run initial equilibration at equilTemp for nstepsEquil
		equilIntegrator = LangevinIntegrator(equilTemp*kelvin, friction/picosecond, tstep*femtoseconds)
		# set up integrator
		equilIntegrator.setConstraintTolerance(0.00001)

		if GPU:
			platform = Platform.getPlatformByName('CUDA')
			equilSimulation = Simulation(psf.topology, system, equilIntegrator, platform, {'CudaPrecision': 'mixed'})
		else:
			platform = Platform.getPlatformByName('CPU')
			equilSimulation = Simulation(psf.topology, system, equilIntegrator, platform, {'Threads': threads})
		equilSimulation.context.setPositions(coor.positions)
		equilSimulation.reporters.append(DCDReporter(outpath+'_equil.dcd', nprint))
		equilSimulation.reporters.append(StateDataReporter(outpath+'_equil.log', nprint, step=True, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, separator='\t'))
		equilSimulation.step(nstepsEquil)
		# now run the quenching portion of the simulation at quenchTemp for nstepsQuench
		finalEquilState = equilSimulation.context.getState(getVelocities=True, getPositions=True)
		del equilSimulation, equilIntegrator
		quenchIntegrator = LangevinIntegrator(quenchTemp*kelvin, friction/picosecond, tstep*femtoseconds)
		quenchIntegrator.setConstraintTolerance(0.00001)
		if GPU:
			platform = Platform.getPlatformByName('CUDA')
			quenchSimulation = Simulation(psf.topology, system, quenchIntegrator, platform, {'CudaPrecision': 'mixed'})
		else:
			platform = Platform.getPlatformByName('CPU')
			quenchSimulation = Simulation(psf.topology, system, quenchIntegrator, platform, {'Threads': threads})
		quenchSimulation.context.setPositions(finalEquilState.getPositions())
		quenchSimulation.reporters.append(DCDReporter(outpath+'_quench_p0.dcd', nprint))
		quenchSimulation.reporters.append(StateDataReporter(outpath+'_quench_p0.log', nprint, step=True, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, separator='\t'))
		quenchSimulation.step(nstepsQuench)

		# write a checkpoint file so we can restart if this run did not fold in the specified run time
		quenchSimulation.saveCheckpoint(outpath+'_quench_p0.chk')

	# if we are restarting a previous run, assume the equilibration phase is complete and we are restarting in the quench phase from the specified checkpoint file
	else:

		quenchIntegrator = LangevinIntegrator(quenchTemp*kelvin, friction/picosecond, tstep*femtoseconds)
		quenchIntegrator.setConstraintTolerance(0.00001)
		if GPU:
			platform = Platform.getPlatformByName('CUDA')
			quenchSimulation = Simulation(psf.topology, system, quenchIntegrator, platform, {'CudaPrecision': 'mixed'})
		else:
			platform = Platform.getPlatformByName('CPU')
			quenchSimulation = Simulation(psf.topology, system, quenchIntegrator, platform, {'Threads': threads})
		# read in the positions and velocities from the specified checkpoint file
		with open (outpath+'_quench_p'+str(resNum-1)+'.chk', 'rb') as f:
			quenchSimulation.context.loadCheckpoint(f.read())
		quenchSimulation.reporters.append(DCDReporter(outpath+'_quench_p'+str(resNum)+'.dcd', nprint))
		quenchSimulation.reporters.append(StateDataReporter(outpath+'_quench_p'+str(resNum)+'.log', nprint, step=True, temperature=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, separator='\t'))
		quenchSimulation.step(nstepsQuench)

		quenchSimulation.saveCheckpoint(outpath+'_quench_p'+str(resNum)+'.chk')

	print ('Temperature quenching complete; data written to '+outpath+'_equil.dcd, '+outpath+'_equil.log, '+outpath+'_quench_p'+str(resNum)+'.dcd, and '+outpath+'_quench_p'+str(resNum)+'.log')

	return

# function to read a file into memory as a list of lines
def readFile (fileName):

        # fileName: path to file to be read into memory
        # returns : file contents as a list of lines

	if os.path.exists(fileName):
	        f_temp = open(fileName)
        	temp   = f_temp.readlines()
        	f_temp.close()
        	return temp
	else:
		print (fileName, 'does not exist.')
		return

"""
# commented out by Dan Nissley 13-Feb-2020
# see note (4) about the issue I was having with adding native contacts once I set this suite of programs up to run as python modules
# function to add native contacts to the forcefield
def addNativeContacts(contacts, system):

        # contacts: a list of lines corresponding to the .contacts file written during model generation

        # returns: nothing, but directly modifies the forcefield object

        # see model generation script for details of these parameters
        k_elec = 332.24*4.184*0.1 # converted to units of (kJ*nm)/(mol*e*e) from units of (kcal*Angstrom)/(mol*e*e),
                                  # computed from Tsai et al. 2015 Protein Science; note that this form lets us treat q1 and q2 as +1, -1, or 0 throughout
        l_d = 1.0                 # units of nm, value for cytoplasm with 100 mM salt at 25 C and eps_r = 80
        eps_r = 80.0              # rough value for water, no units

        nativeContacts = CustomBondForce('(eps*((13.*(sigma/r)^12.)+(-18.*(sigma/r)^10.)+(4.*(sigma/r)^6.))) + (k_elec*((q1q2)/(eps_r*r))*exp(-r/l_d))')
        nativeContacts.addPerBondParameter('eps')
        nativeContacts.addPerBondParameter('sigma')
        nativeContacts.addPerBondParameter('q1q2') # the product of the unit charges of the beads involved
        nativeContacts.addGlobalParameter('k_elec', k_elec)
        nativeContacts.addGlobalParameter('eps_r', eps_r)
        nativeContacts.addGlobalParameter('l_d', l_d)
        pairsToExclude = [] # make a list of pairs of residues between which a native contact will be formed - will be excluded from repulsive LJ term
        system.addForce(nativeContacts)
        for i in range (0, len(contacts)):
                r1, r2, sigma, eps, q1q2 = contacts[i].split()[0:5]
                nativeContacts.addBond(int(r1)-1, int(r2)-1, [np.float64(eps), np.float64(sigma), np.float64(q1q2)])
                pairsToExclude.append([int(r1)-1, int(r2)-1])

        # need to make sure we update the exclusion list so that all pairs of residues that share a native contact are on the exclusion list
        # for the purely repulsive nonbonded term - this makes sure the NN and NC terms are calculated separately.

        # make a list of the current exclusions in the repulsive custom nonbonded force
        repulsiveNBforce = system.getForce(3) # note that the force index depends on the order in which they are listed in the forcefield.xml file, indexed as [0, 1, 2, 3, ...]
        currExclusions = []
        for i in range (0, repulsiveNBforce.getNumExclusions()):
                currExclusions.append(repulsiveNBforce.getExclusionParticles(i))
        # add pairs of residues not already in exclusion list to the exclusion list
        for pair in pairsToExclude:
                if pair not in currExclusions:
                        repulsiveNBforce.addExclusion(pair[0], pair[1])

        return
"""

# function that profiles potential energy of each force field term
def profileEnergies(system, simulation):

	forceGroups = getForceGroups(system)
	energies = getEnergyDecomposition(simulation.context, forceGroups)
	e = 0
	for forceGroup in energies:
		if e == 0:
			totalEnergy = energies[forceGroup]
		else:
			totalEnergy += energies[forceGroup]
		print (forceGroup, energies[forceGroup])
		e += 1
	print ('Current total potential energy is:', totalEnergy)

	return

# function to get a dictionary of force groups from a system object
def getForceGroups(system):

	# system: an OpenMM System object

	# returns: a list of the force groups in the system

	forcegroups = {}
	for i in range (0, system.getNumForces()):
		force = system.getForce(i)
		force.setForceGroup(i)
		forcegroups[force] = i
	return forcegroups

# function to compute energy from each force group in a context
def getEnergyDecomposition(context, forcegroups):

	# context: the OpenMM context
	# forcegroups: a list of forcegroups

	# returns: a list of energies

	energies = {}
	for f, i in forcegroups.items():
		energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
	return energies


"""
# commented out by Dan Nissley 12-Feb-2020 - use the new version! It correctly handles electrostatics
# function to add native contacts to the forcefield
def OLDaddNativeContacts(contacts):

	# contacts: a list of lines corresponding to the .contacts file written during model generation

	# returns: nothing, but directly modifies the forcefield object for this simulation

	# note: need to add in the electrostatic portion of the forcefield term again, but ignore it for now

	nativeContacts = CustomBondForce('eps*((13.*(sig/r)^12.)+(-18.*(sig/r)^10.)+(4.*(sig/r)^6.))')
	nativeContacts.addPerBondParameter('eps')
	nativeContacts.addPerBondParameter('sig')
	pairsToExclude = [] # make a list of pairs of residues between which a native contact will be formed - will be excluded from repulsive LJ term
	system.addForce(nativeContacts)
	for i in range (0, len(contacts)):
		r1, r2, d, eps = contacts[i].split()[0:4]
		nativeContacts.addBond(int(r1)-1, int(r2)-1, [np.float64(eps), np.float64(d)])
		pairsToExclude.append([int(r1)-1, int(r2)-1])
	repulsiveNBforce = system.getForce(3)

	# make a list of the current exclusions in the repulsive custom nonbonded force
	currExclusions = []
	for i in range (0, repulsiveNBforce.getNumExclusions()):
		currExclusions.append(repulsiveNBforce.getExclusionParticles(i))
	# add pairs of residues not already in exclusion list to the exclusion list
	for pair in pairsToExclude:
		if pair not in currExclusions:
			repulsiveNBforce.addExclusion(pair[0], pair[1])
"""
