from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from datetime import datetime # ADDED

pdb = PDBFile('MD_practicals/1_practical/input.pdb') # MODIFIED
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)

# integrator setup
step_size = 0.01 # 0.004 originally
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, step_size*picoseconds)

# simulation setup
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

output_pdbpath = 'MD_practicals/1_practical/output2.pdb'
output_frequency = 1 # save every n time steps
simulation.reporters.append(PDBReporter(output_pdbpath, output_frequency)) # MODIFIED
simulation.reporters.append(StateDataReporter(stdout, output_frequency, step=True,
        potentialEnergy=True, temperature=True)) # MODIFIED

startTime = datetime.now() # ADDED

# simulation production
N_steps = 10000 # simulation steps
simulation.step(N_steps) # COMMENT: sim time step

print('Execution time:', datetime.now()-startTime) # ADDED
