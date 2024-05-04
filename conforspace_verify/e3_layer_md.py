#!/usr/bin/env python
import os 
import numpy as np
import sys
import h5py
import torch
from torchmd.integrator import maxwell_boltzmann
from torchmd.systems import System
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.utils import LogWriter
from tqdm import tqdm
from e3_layers.utils import build
from e3_layers import configs
from e3_layers.data import Batch, computeEdgeIndex
import ase
import os


langevin_temperature = int(sys.argv[1]); steps = int(sys.argv[2]); output_period = int(sys.argv[3])

class Parameters:
    def __init__(self, masses, mapped_atom_type, precision=torch.float, device="cpu"):
        self.masses = masses 
        self.mapped_atom_type = mapped_atom_type
        self.natoms = len(masses)
        
        self.build_parameters()
        self.precision_(precision)
        self.to_(device)

    def to_(self, device):
        self.masses = self.masses.to(device)
        self.device = device

    def precision_(self, precision):
        self.masses = self.masses.type(precision)

    def build_parameters(self):
        uqatomtypes, indexes = np.unique(self.mapped_atom_type, return_inverse=True)
        self.mapped_atom_types = torch.tensor(indexes)
        masses = torch.tensor(self.masses)
        masses.unsqueeze_(1)
        self.masses = masses

class Myclass():
    def __init__(self, config, atom_types, parameters, r_max=None):
        # information such as masses, used by the integrator
        self.par = parameters
        self.atom_types = atom_types
        self.model = build(config).to(device)
        self.n_nodes = torch.ones((1, 1), dtype=torch.long)* atom_types.shape[0]
        if r_max is None:
            self.r_max = config.r_max
        else:
            self.r_max = r_max

    def compute(self, pos, box, forces):
        data = {'pos': pos[0], 'species': self.atom_types, '_n_nodes': self.n_nodes}
        attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
        _data, _attrs = computeEdgeIndex(data, attrs, r_max=self.r_max)
        data.update(_data)
        attrs.update(_attrs)
        batch = Batch(attrs, **data).to(device)
        batch = self.model(batch)
        forces[0, :] = batch['forces'].detach() * 23.0605426
        return [batch['energy'].item()]


precision = torch.float
device = "cpu"
#device = "cuda:0"

mass = {'H':1.00794,'C':12.0107,'N':14.0067,'O':15.9994,'S':32.065}
sym_dict = {'H':1,'C':6,'N':7,'O':8,'S':16}
lmp_map = ["C","H","N","O","S"]; flag = False
lmp_type = []

mol_coords = []; mol_atypes = []; mol_masses = []
with open('input.xyz','r') as fp:
    lines = fp.readlines()[1:]
    for line in lines:
        line = line.strip().split()
        if len(line) == 4:
            mol_coords.append([float(x) for x in line[1:4]])
            mol_atypes.append(sym_dict[line[0]])
            mol_masses.append(mass[line[0]])
            lmp_type.append(line[0])

mol_atypes = np.array(mol_atypes,dtype=np.int64)
mol_masses = np.array(mol_masses)
mol_numAtoms = len(mol_masses)
mol_coords = np.array(mol_coords).reshape((mol_numAtoms,3,1))
parameters = Parameters(mol_masses, mol_atypes, precision=precision, device=device)

system = System(mol_numAtoms, nreplicas=1, precision=precision, device=device)
system.set_positions(mol_coords)
system.set_box(np.zeros((3,1)))
system.set_velocities(maxwell_boltzmann(parameters.masses, T=langevin_temperature, replicas=1))

config = configs.config_energy_force().model_config
#config.n_dim = 32
atom_types = torch.tensor((mol_atypes))
forces = Myclass(config, atom_types, parameters)
state_dict = torch.load('../best.pt', map_location=device)
model_state_dict = {}
for key, value in state_dict.items():
    if key[:7] == 'module.':
        key = key[7:]
    model_state_dict[key] = value
forces.model.load_state_dict(model_state_dict)


langevin_gamma = 0.1
timestep = 1

integrator = Integrator(system, forces, timestep, device, gamma=langevin_gamma, T=langevin_temperature)
wrapper = Wrapper(mol_numAtoms, None, device)

logger = LogWriter(path="./", keys=('iter','ns','epot','ekin','etot','T'), name='monitor.csv')

FS2NS = 1E-6
save_period = output_period; traj = []

trajectroyout = "traj.npy"


iterator = tqdm(range(1, int(steps / output_period) + 1))
Epot = forces.compute(system.pos, system.box, system.forces)
for i in iterator:
    Ekin, Epot, T = integrator.step(niter=output_period)
    wrapper.wrap(system.pos, system.box)
    currpos = system.pos.detach().cpu().numpy().copy()
    traj.append(currpos)
    if (i*output_period) % save_period == 0:
        np.save(trajectroyout, np.stack(traj, axis=2))

    logger.write_row({'iter':i*output_period,'ns':FS2NS*i*output_period*timestep,'epot':Epot,'ekin':Ekin,'etot':Epot+Ekin,'T':T})

coords = np.transpose(np.load('traj.npy')[0],(1,0,2))
atomic_numbers = []
for ii in range(len(coords)):
    atomic_numbers.append(mol_atypes)

np.savez_compressed('traj.npz', coords=coords, atomic_numbers=atomic_numbers)


