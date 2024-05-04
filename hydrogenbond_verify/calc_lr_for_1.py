#!/usr/bin/env python 
import numpy as np
import pickle
import os
from dmff_torch.multipole import rot_global2local, rot_local2global, C1_c2h, C2_c2h
import ase 

atomic_num = ase.data.atomic_numbers

def convert_tensor(mom):
    return torch.tensor(np.array(mom),dtype=torch.float32)

def irreps2matrix(qua,device='cuda:0'):
    basis = o3.wigner_3j(1, 1, 2, device=device)
    # multiple 15. or 5. should be check
    result = basis@qua * torch.tensor(15.,dtype=torch.float32)
    result = result.view(9,result.shape[2])[[0,4,8,1,2,5]]
    return result

def get_axis_idx(ii,conn_atom,ele):
    ele = [atomic_num[u] for u in ele]
    z_idx = None; x_idx = None
    nei_0 = conn_atom[ii]
    if len(nei_0) == 1:
        z_idx = nei_0[0]
        nei_1 = conn_atom[z_idx]
        nei_ele = np.array([ele[u] for u in nei_1])
        nei_1 = np.array(nei_1)[np.argsort(-nei_ele)]
        for uu in nei_1:
            if uu != ii and x_idx == None:
                x_idx = uu
    else:
        nei_ele = [ele[u] for u in nei_0]
        z_idx = nei_0[np.argsort(nei_ele)[-1]]
        x_idx = nei_0[np.argsort(nei_ele)[-2]]
    assert(z_idx != None and x_idx !=None)
    return z_idx, x_idx

def check_topo(coord, topo, symbol):
    # check whether topo is reasonable, charge transfer may happen
    state = True; coord = np.array(coord)
    topo_0 = topo[0]; topo_1 = topo[1] + np.max(topo_0) + 1
    for bond in topo_0:
        sym_0 = symbol[bond[0]]; sym_1 = symbol[bond[1]]
        c_0 = coord[bond[0]]; c_1 = coord[bond[1]]
        if sym_0 == 'H' or sym_1 == 'H':
            dis = np.linalg.norm((c_0 - c_1))
            if dis > 2.0:
                state = False            
    for bond in topo_1:
        sym_0 = symbol[bond[0]]; sym_1 = symbol[bond[1]]
        c_0 = coord[bond[0]]; c_1 = coord[bond[1]]
        if sym_0 == 'H' or sym_1 == 'H':
            dis = np.linalg.norm((c_0 - c_1))
            if dis > 2.0:
                state = False
    return state 

def input_infor(topo, mol_num, coord, symbol):
    topo_0 = topo[0]; topo_1 = topo[1]
    topo = np.vstack((np.array(topo_0), np.array(topo_1) + mol_num[0]))
    
    symbol_0 = symbol[0:mol_num[0]]; symbol_1 = symbol[mol_num[0]:]
    axis_types, axis_indices = init_axis(topo, symbol)
    axis_types_0, axis_indices_0 = init_axis(topo_0, symbol_0)
    axis_types_1, axis_indices_1 = init_axis(topo_1, symbol_1)
    coord_0 = coord[0:mol_num[0]]; coord_1 = coord[mol_num[0]:]
    #coord = convert_tensor(coord); coord_0 = convert_tensor(coord_0)
    #coord_1 = convert_tensor(coord_1) 
    axis_types = torch.tensor(axis_types); axis_indices = torch.tensor(axis_indices)
    axis_types_0 = torch.tensor(axis_types_0); axis_indices_0 = torch.tensor(axis_indices_0)
    axis_types_1 = torch.tensor(axis_types_1); axis_indices_1 = torch.tensor(axis_indices_1)
    return coord, coord_0, coord_1, axis_types, axis_types_0, axis_types_1, axis_indices, axis_indices_0, axis_indices_1, \
           topo, topo_0, topo_1

def init_axis(topo,symbol):
    #topo = topo.tolist()
    conn_atom = {}
    for pair in topo:
        conn_atom[pair[0]] = []
        conn_atom[pair[1]] = []
    for pair in topo:
        conn_atom[pair[0]].append(pair[1])
        conn_atom[pair[1]].append(pair[0])
    
    for ii in range(len(symbol)):
        conn_atom[ii] = list(set(conn_atom[ii]))

    axis_types = []; axis_indices = []; ZThenX = 0; yaxis=-1
    for ii in range(len(symbol)):
        axis_types.append(ZThenX)
        zaxis, xaxis = get_axis_idx(ii, conn_atom, symbol)
        axis_indices.append([zaxis,xaxis,yaxis])
    axis_types = np.array(axis_types); axis_indices = np.array(axis_indices)
    return axis_types,axis_indices

def gen_pair(coord, topo):
    # attention build_covalent_map may not suitable
    data = {'positions':coord, 'bonds':topo}
    cov_map = build_covalent_map(data, 6)
    pairs = []
    for na in range(len(coord)):
        for nb in range(na + 1, len(coord)):
            pairs.append([na, nb, 0])
    pairs = np.array(pairs, dtype=int)
    pairs[:,2] = cov_map[pairs[:,0], pairs[:,1]]
    return torch.tensor(pairs)

def pmepol(box, axis_types, axis_indices, rcut, coord, pairs, q_local, pol, tholes, mscales, pscales, dscales):
    pme_es_pol = ADMPPmeForce(box, axis_types, axis_indices, rcut, 5e-4, 2, lpol=True, lpme=False, steps_pol=5)
    #U_ind = pme_es_pol.U_ind
    e_es_pol = pme_es_pol.get_energy(coord, box, pairs, q_local, pol, tholes, mscales, pscales, dscales, None, False)
    U_ind = pme_es_pol.U_ind
    return e_es_pol, U_ind

def pme(box, coord, pairs, q_local, U_ind, pol, tholes, mscales, pscales, dscales, construct_local_frame):
    e = energy_pme(coord, box, pairs, q_local, U_ind, pol, tholes, mscales, pscales, dscales, None, 0, None, None, None, 2, True, None, None, None, False, lpme=False)
    return e

def load_model(config_name,f_path,device):
    config = config_name.model_config
    model = build(config).to(device)
    state_dict = torch.load(f_path, map_location=device)
    model_state_dict = {}
    for key, value in state_dict.items():
        if key[:7] == 'module.':
            key = key[7:]
        model_state_dict[key] = value
    model.load_state_dict(model_state_dict)
    return model

def load_input(coord, atom_type, n_nodes,r_cutnn,device):
    data = {'pos': coord, 'species': atom_type, '_n_nodes': n_nodes}
    attrs = {'pos': ('node', '1x1o'), 'species': ('node','1x0e')}
    _data, _attrs = computeEdgeIndex(data, attrs, r_max=r_cutnn)
    data.update(_data)
    attrs.update(_attrs)
    input_batch = Batch(attrs, **data).to(device)
    return input_batch

def convert_e3nn(q,dip,qua,pol,device_0):
    dip = torch.matmul(C1_c2h,dip.T).T
    qua = irreps2matrix(qua.T,device=device_0)
    qua = torch.matmul(C2_c2h,qua).T
    pol = pol.squeeze()
    q_global = torch.hstack((q, dip, qua))
    #c_list = torch.hstack((c6, c8, c10))
    return q_global, pol

# shift along the center of mass direction
def find_closest_distance(coord_A, coord_B):
    coord_A = np.array(coord_A); coord_B = np.array(coord_B)
    n_atoms1 = len(coord_A); n_atoms2 = len(coord_B)
    min_i = -1; min_j = -1
    min_dr = 10000
    for i in range(n_atoms1):
        r1 = coord_A[i]
        for j in range(n_atoms2):
            r2 = coord_B[j]
            if np.linalg.norm(r1-r2) < min_dr:
                min_dr = np.linalg.norm(r1-r2)
                min_i = i
                min_j = n_atoms1 + j
    return min_i, min_j, min_dr

# sqrt the c6, c8, and c10
def sqrt_monopole(disp_coeff):
    disp_coeff = torch.sqrt(torch.clamp(disp_coeff, min=0.))
    return disp_coeff

def read_data(f_dir):
    coord_A = []; symbol_A = []; coord_B = []; symbol_B = []
    elec = None; Ind = None; Ind_hf = None; Ind_mp2 = None; Disp = None  
    q_net_A = None; q_net_B = None 
    f_inp = os.path.join(f_dir, 'input.inp')
    with open(f_inp, 'r') as fp:
        lines = fp.readlines()
        read_a = True
        for line in lines:
            line = line.strip().split()
            if len(line) == 1:
                if line[0] == '--':
                    read_a = False 
            if read_a == True:
                if len(line) == 2:
                    if line[-1] == '1':
                        q_net_A = int(line[0])
                elif len(line) == 4:
                    coord_A.append([float(u) for u in line[1:]])
                    symbol_A.append(line[0])
            else:
                if len(line) == 2:
                    if line[-1] == '1':
                        q_net_B = int(line[0])
                elif len(line) == 4:
                    coord_B.append([float(u) for u in line[1:]])
                    symbol_B.append(line[0])
    f_out = os.path.join(f_dir, 'input.inp.dat')
    with open(f_out, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) == 7:
                if line[0] == 'Electrostatics':
                    elec = float(line[3])
                elif line[0] == 'Induction':
                    Ind = float(line[3])
                elif line[0] == 'Dispersion':
                    Disp = float(line[3])
            if len(line) == 9:
                if line[0] == 'delta' and line[1] == 'HF,r':
                    Ind_hf = float(line[5])
                elif line[0] == 'delta' and line[1] == 'MP2,r':
                    Ind_mp2 = float(line[5])
    return coord_A, symbol_A, coord_B, symbol_B, elec, Ind, Ind_hf, Ind_mp2, Disp, q_net_A, q_net_B

def gencom(coord, symbol, idx):
    with open(f'frame_{idx}.com','w') as fp:
        fp.write('# pm6 \n')
        fp.write('\n')
        fp.write('title\n')
        fp.write('\n')
        fp.write('0 1 \n')
        for ss, cc in zip(symbol, coord):
            fp.write(f'{ss} {cc[0]} {cc[1]} {cc[2]} \n')
        fp.write('\n')
    return 


if __name__ == '__main__':
    opera = 'calc_lr'
    if opera == 'collect_sapt':
        from glob import glob
        # first, read the sapt
        f_dirs = glob('./conf_*/mol_*/input.inp.dat')
        coords_A = []; symbols_A = []; coords_B = []; symbols_B = []
        E_elec = []; E_ind = []; E_ind_hf = []; E_ind_mp2 = []; E_disp = []
        q_net_A = []; q_net_B = []; confs = [] 
        for f_dir in f_dirs:
            f_dir = os.path.dirname(f_dir)
            coord_A, symbol_A, coord_B, symbol_B, elec, Ind, Ind_hf,\
                           Ind_mp2, Disp, q_A, q_B = read_data(f_dir)
            coords_A.append(coord_A); symbols_A.append(symbol_A)
            coords_B.append(coord_B); symbols_B.append(symbol_B)
            E_elec.append(elec); E_ind.append(Ind); E_ind_hf.append(Ind_hf)
            E_ind_mp2.append(Ind_mp2); E_disp.append(Disp)
            q_net_A.append(q_A); q_net_B.append(q_B)
            confs.append(os.path.basename(os.path.dirname(f_dir)))

        np.savez('sapt.npz', coord_A=coords_A, symbol_A=symbols_A, coord_B=coords_B, symbol_B=symbols_B, \
                E1pol=E_elec, E2ind=E_ind, E2disp=E_disp, q_net_A=q_net_A, q_net_B=q_net_B, E_ind_hf=E_ind_hf,\
                E_ind_mp2=E_ind_mp2, conf=confs)
                
    elif opera == 'calc_param':
        import torch 
        import ase
        from e3_layers.utils import build
        from e3_layers import configs
        from e3_layers.data import Batch, computeEdgeIndex
        from dmff_torch.recip import pme_recip
        from e3nn import o3
        
        atomic_num = ase.data.atomic_numbers
        # we may save all the param in harmonic format 
        data_dimer = dict(np.load('sapt.npz',allow_pickle=True))
        data_frg = np.load('frag.npz',allow_pickle=True)
        confs = data_dimer['conf']
        symbols_frg = data_frg['symbol']
        bonds_frg = data_frg['bond']
        coords_dimer = data_dimer['coord_A']; symbols_dimer = data_dimer['symbol_A']
        coords_trimer = data_dimer['coord_B']; symbols_trimer = data_dimer['symbol_B']
        
        # now using nn to predict the tensor, we will save the data as harmonic format
        device = 'cpu'; r_cutnn = 5.; rcut = 10.
        model_q = load_model(configs.config_monopole(), './e3nnmodel/q.pt', device)
        model_dip = load_model(configs.config_dipole(), './e3nnmodel/dipole.pt', device)
        model_qua = load_model(configs.config_quadrupole(), './e3nnmodel/qua.pt', device)
        model_pol = load_model(configs.config_monopole(), './e3nnmodel/pol.pt', device)
        model_c6 = load_model(configs.config_monopole(), './e3nnmodel/c6.pt', device)
        model_c8 = load_model(configs.config_monopole(), './e3nnmodel/c8.pt', device)
        model_c10 = load_model(configs.config_monopole(), './e3nnmodel/c10.pt', device)
        # predict the q, dip, qua and pol 
        Q_0 = []; Q_1 = []; Q_2 = []; Dip_0 = []; Dip_1 = []; Dip_2 = []
        Qua_0 = []; Qua_1 = []; Qua_2 = []; Pol_0 = []; Pol_1 = []; Pol_2 = []
        Q_global_0 = []; Q_global_1 = []; Q_global_2 = []
        Dis = []; Topo_0 = []; Topo_1 = []; Topo_2 = [] 
        C_list_0 = []; C_list_1 = []; C_list_2 = []
        # we need the confs, so we know three monomers 
        for idx, conf in enumerate(confs):
            print(f'**********{idx}')
            conf_0 = int(conf.split('_')[1]); conf_1 = int(conf.split('_')[2])
            conf_2 = int(conf.split('_')[3])
            topo_0 = bonds_frg[conf_0]; topo_1 = bonds_frg[conf_1]; topo_2 = bonds_frg[conf_2]

            n_atom_0 = len(symbols_frg[int(conf_0)])
            n_atom_1 = len(symbols_frg[int(conf_1)])
            
            coord_0 = coords_dimer[idx][0:n_atom_0]; coord_1 = coords_dimer[idx][n_atom_0:]
            symbol_0 = symbols_dimer[idx][0:n_atom_0]; symbol_1 = symbols_dimer[idx][n_atom_0:]
            coord_2 = coords_trimer[idx]; symbol_2 = symbols_trimer[idx]        
            dis_min = find_closest_distance(coords_dimer[idx], coords_trimer[idx])
            species_0 = torch.tensor([atomic_num[u] for u in symbol_0],dtype=torch.long)
            species_1 = torch.tensor([atomic_num[u] for u in symbol_1],dtype=torch.long)
            species_2 = torch.tensor([atomic_num[u] for u in symbol_2],dtype=torch.long)
            n_nodes_0 = torch.ones((1, 1), dtype=torch.long)* len(coord_0)
            n_nodes_1 = torch.ones((1, 1), dtype=torch.long)* len(coord_1)
            n_nodes_2 = torch.ones((1, 1), dtype=torch.long)* len(coord_2)
            input_0 = load_input(coord_0, species_0, n_nodes_0, r_cutnn, device)
            input_1 = load_input(coord_1, species_1, n_nodes_1, r_cutnn, device)
            input_2 = load_input(coord_2, species_2, n_nodes_2, r_cutnn, device)
            q_0 = model_q(input_0)['monopole']; dip_0 = model_dip(input_0)['dipole']
            qua_0 = model_qua(input_0)['quadrupole_2']; pol_0 = model_pol(input_0)['monopole']
            q_1 = model_q(input_1)['monopole']; dip_1 = model_dip(input_1)['dipole']
            qua_1 = model_qua(input_1)['quadrupole_2']; pol_1 = model_pol(input_1)['monopole']
            q_2 = model_q(input_2)['monopole']; dip_2 = model_dip(input_2)['dipole']
            qua_2 = model_qua(input_2)['quadrupole_2']; pol_2 = model_pol(input_2)['monopole']
            c6_0 = model_c6(input_0)['monopole']; c8_0 = model_c8(input_0)['monopole']
            c10_0 = model_c10(input_0)['monopole']
            c6_1 = model_c6(input_1)['monopole']; c8_1 = model_c8(input_1)['monopole']
            c10_1 = model_c10(input_1)['monopole']
            c6_2 = model_c6(input_2)['monopole']; c8_2 = model_c8(input_2)['monopole']
            c10_2 = model_c10(input_2)['monopole']
            c6_0 = sqrt_monopole(c6_0); c8_0 = sqrt_monopole(c8_0); c10_0 = sqrt_monopole(c10_0)
            c6_1 = sqrt_monopole(c6_1); c8_1 = sqrt_monopole(c8_1); c10_1 = sqrt_monopole(c10_1)
            c6_2 = sqrt_monopole(c6_2); c8_2 = sqrt_monopole(c8_2); c10_2 = sqrt_monopole(c10_2)
            c_list_0 = torch.hstack((c6_0, c8_0, c10_0))
            c_list_1 = torch.hstack((c6_1, c8_1, c10_1))
            c_list_2 = torch.hstack((c6_2, c8_2, c10_2))
            q_global_0, pol_0 = convert_e3nn(q_0, dip_0, qua_0, pol_0, device)
            q_global_1, pol_1 = convert_e3nn(q_1, dip_1, qua_1, pol_1, device)
            q_global_2, pol_2 = convert_e3nn(q_2, dip_2, qua_2, pol_2, device)
            Q_0.append(q_0.tolist()); Q_1.append(q_1.tolist()); Q_2.append(q_2.tolist())
            Dip_0.append(dip_0.tolist()); Dip_1.append(dip_1.tolist()); Dip_2.append(dip_2.tolist())
            Qua_0.append(qua_0.tolist()); Qua_1.append(qua_1.tolist()); Qua_2.append(qua_2.tolist())
            Pol_0.append(pol_0.tolist()); Pol_1.append(pol_1.tolist()); Pol_2.append(pol_2.tolist())
            Q_global_0.append(q_global_0.tolist()); Q_global_1.append(q_global_1.tolist()); Q_global_2.append(q_global_2.tolist())
            Dis.append(dis_min); Topo_0.append(topo_0); Topo_1.append(topo_1); Topo_2.append(topo_2)
            C_list_0.append(c_list_0.tolist()); C_list_1.append(c_list_1.tolist()); C_list_2.append(c_list_2.tolist())

        data_dimer['q_0'] = Q_0; data_dimer['q_1'] = Q_1; data_dimer['q_2'] = Q_2
        data_dimer['dip_0'] = Dip_0; data_dimer['dip_1'] = Dip_1; data_dimer['dip_2'] = Dip_2
        data_dimer['qua_0'] = Qua_0; data_dimer['qua_1'] = Qua_1; data_dimer['qua_2'] = Qua_2
        data_dimer['pol_0'] = Pol_0; data_dimer['pol_1'] = Pol_1; data_dimer['pol_2'] = Pol_2
        data_dimer['q_global_0'] = Q_global_0; data_dimer['q_global_1'] = Q_global_1; data_dimer['q_global_2'] = Q_global_2
        data_dimer['dis'] = Dis; data_dimer['topo_0'] = Topo_0; data_dimer['topo_1'] = Topo_1; data_dimer['topo_2'] = Topo_2
        data_dimer['c_list_0'] = C_list_0; data_dimer['c_list_1'] = C_list_1
        data_dimer['c_list_2'] = C_list_2
        np.savez('sapt_nn.npz', **data_dimer)
        

    elif opera == 'calc_lr':
        import torch
        from dmff_torch.pairwise import (generate_pairwise_interaction,
                            TT_damping_qq_c6_kernel,
                            TT_damping_qq_kernel,
                            slater_disp_damping_kernel,
                            slater_sr_kernel,
                            distribute_scalar,
                            distribute_multipoles,
                            distribute_v3, 
                            distribute_dispcoeff)
        from dmff_torch.nblist import build_covalent_map
        from dmff_torch.disp import energy_disp_pme
        from dmff_torch.pme import ADMPPmeForce, energy_pme, setup_ewald_parameters
        from functools import partial
        from dmff_torch.utils import regularize_pairs, pair_buffer_scales
        from dmff_torch.spatial import pbc_shift, v_pbc_shift, generate_construct_local_frames
        from dmff_torch.multipole import rot_global2local, rot_local2global, C1_c2h, C2_c2h
        import ase
        from e3_layers.utils import build
        from e3_layers import configs
        from e3_layers.data import Batch, computeEdgeIndex
        from dmff_torch.recip import pme_recip
        from e3nn import o3
        # read the topo, coord and energy components
        data = dict(np.load('sapt_nn.npz',allow_pickle=True))
        coords_A = data['coord_A']; symbols_A = data['symbol_A']
        coords_B = data['coord_B']; symbols_B = data['symbol_B']
        Q_0 = data['q_global_0']; Q_1 = data['q_global_1']; Q_2 = data['q_global_2']
        Pol_0 = data['pol_0']; Pol_1 = data['pol_1']; Pol_2 = data['pol_2']
        Topo_0 = data['topo_0']; Topo_1 = data['topo_1']; Topo_2 = data['topo_2']
        C_list_0 = data['c_list_0']; C_list_1 = data['c_list_1']; C_list_2 = data['c_list_2']

        box = torch.tensor([[50., 0., 0.], [0.,50.,0.],[0.,0.,50.]], dtype=torch.float32, requires_grad=False)
        rcut = 10.

        E_es_tot = []; E_pol_tot = []; E_disp_tot = []
        for ii in range(len(Q_0)):
            coord = np.concatenate((coords_A[ii], coords_B[ii]))
            symbol = np.concatenate((symbols_A[ii], symbols_B[ii]))
            coord = torch.tensor(coord, dtype=torch.float32)
            mol_num = [len(Q_0[ii]),len(Q_1[ii]),len(Q_2[ii])]
            species = torch.tensor([atomic_num[u] for u in symbol],dtype=torch.long)
            species_A = species[0:mol_num[0]]; species_B = species[mol_num[0]:]
            q_0 = torch.tensor(Q_0[ii],dtype=torch.float32); q_1 = torch.tensor(Q_1[ii],dtype=torch.float32)
            q_2 = torch.tensor(Q_2[ii],dtype=torch.float32)
            pol_0 = torch.tensor(Pol_0[ii],dtype=torch.float32); pol_1 = torch.tensor(Pol_1[ii],dtype=torch.float32)
            pol_2 = torch.tensor(Pol_2[ii],dtype=torch.float32)
            q_global = torch.vstack((q_0, q_1, q_2))
            pol = torch.hstack((pol_0, pol_1, pol_2))
            q_global_A = torch.vstack((q_0, q_1))
            q_global_B = q_2
            pol_A = torch.hstack((pol_0, pol_1))
            pol_B = pol_2 

            c_list_0 = torch.tensor(C_list_0[ii],dtype=torch.float32)
            c_list_1 = torch.tensor(C_list_1[ii],dtype=torch.float32)
            c_list_2 = torch.tensor(C_list_2[ii],dtype=torch.float32)

            c_list_A = torch.vstack((c_list_0, c_list_1))
            c_list_AB = torch.vstack((c_list_A, c_list_2))
            c_list_B = c_list_2

            topo_0 = np.array(Topo_0[ii])
            topo_1 = np.array(Topo_1[ii]) + mol_num[0]
            topo_2 = np.array(Topo_2[ii])
            
            topo_0 = topo_0[:,:2]; topo_1 = topo_1[:,:2]; topo_2 = topo_2[:,:2]
            topo_A = np.vstack((topo_0, topo_1)); topo_B = topo_2
            
            topo = [topo_A, topo_B]
            #if ii == 0:
            #    gencom(coord, symbol, 0)
            #print(topo)
            #print(np.array(topo[-1]) + len(pol_A))
            coord, coord_A, coord_B, axis_types, axis_types_A, axis_types_B, axis_indices, axis_indices_A, axis_indices_B, \
                    topo_t, topo_A, topo_B = input_infor(topo, [mol_num[0] + mol_num[1], mol_num[2]], coord, symbol)

            Q_local = q_global
            Q_local_A = Q_local[0:len(q_global_A)] 
            Q_local_B = Q_local[len(q_global_A):]

            # get the pairs, tholes, mscales, pscales, dscales
            _tholes = [0.33] * len(coord)
            _tholes_A = _tholes[0:len(q_global_A)]; _tholes_B = _tholes[len(q_global_A):]
            _tholes = convert_tensor(_tholes); _tholes_A = convert_tensor(_tholes_A); _tholes_B = convert_tensor(_tholes_B)
            mScales = torch.tensor([0.,0.,0.,0.,0.,1.])
            pScales = torch.tensor([0.,0.,0.,0.,0.,1.])
            dScales = torch.tensor([0.,0.,0.,0.,0.,1.])

            # gen pairs 
            pairs = gen_pair(coord, topo_t)
            pairs_A = gen_pair(coord_A, topo_A)
            pairs_B = gen_pair(coord_B, topo_B)

            #################################
            # electrostatic + pol
            #################################
            e_es_pol_AB, U_ind_AB = pmepol(box, axis_types, None, rcut, coord, pairs, Q_local, pol, _tholes, mScales, pScales, dScales)
            e_es_pol_A, U_ind_A = pmepol(box, axis_types_A, None, rcut, coord_A, pairs_A, Q_local_A, pol_A, _tholes_A, mScales, pScales, dScales)
            e_es_pol_B, U_ind_B = pmepol(box, axis_types_B, None, rcut, coord_B, pairs_B, Q_local_B, pol_B, _tholes_B, mScales, pScales, dScales)
            E_espol = e_es_pol_AB - e_es_pol_A - e_es_pol_B

            #################################
            # polarization (induction) energy
            #################################
            U_ind_AB_mono = torch.vstack((U_ind_A, U_ind_B))
            e_AB_nonpol = pme(box, coord, pairs, Q_local, U_ind_AB_mono, pol, _tholes, mScales, pScales, dScales, None)
            e_A_nonpol = pme(box, coord_A, pairs_A, Q_local_A, U_ind_A, pol_A, _tholes_A, mScales, pScales, dScales, None)
            e_B_nonpol = pme(box, coord_B, pairs_B, Q_local_B, U_ind_B, pol_B, _tholes_B, mScales, pScales, dScales, None)
            E_es = e_AB_nonpol - e_A_nonpol - e_B_nonpol
            E_pol = E_espol - E_es

            e_disp_AB= energy_disp_pme(coord, box, pairs, c_list_AB, mScales, None, None, None, None, 10, \
                        None, None, None, None, None, None, None, None, False, lpme=False)
            e_disp_A = energy_disp_pme(coord_A, box, pairs_A, c_list_A, mScales, None, None, None, None, 10, \
                        None, None, None, None, None, None, None, None, False, lpme=False)
            e_disp_B = energy_disp_pme(coord_B, box, pairs_B, c_list_B, mScales, None, None, None, None, 10, \
                        None, None, None, None, None, None, None, None, False, lpme=False)
            E_disp = e_disp_AB - e_disp_A - e_disp_B

            E_es = E_es / 4.184; E_pol = E_pol / 4.184; E_disp = - E_disp * 627.509
            E_es_tot.append(E_es.tolist()); E_pol_tot.append(E_pol.tolist())
            E_disp_tot.append(E_disp.tolist())
        data['E_es'] = E_es_tot; data['E_pol'] = E_pol_tot; data['E_disp'] = E_disp_tot
        np.savez('sapt_nn.npz', **data)
            
        
