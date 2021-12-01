from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from datetime import datetime # ADDED


pdb = PDBFile('MD_practicals/1_practical/input.pdb') # MODIFIED
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(PDBReporter('MD_practicals/1_practical/output.pdb', 100)) # MODIFIED
simulation.reporters.append(StateDataReporter(stdout, 100, step=True,
        potentialEnergy=True, temperature=True)) # MODIFIED
startTime = datetime.now() # ADDED
simulation.step(10000) # COMMENT: sim time steps
print('Execution time:', datetime.now()-startTime) # ADDED
