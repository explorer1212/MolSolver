import os
from rdkit import Chem
import torch
import glob
import pandas as pd
import numpy as np
import contextlib
from tqdm import tqdm
from posecheck import PoseCheck
from rdkit.Chem.QED import qed
import sys
sys.path.append("/data/mntdata/aaa/molsolver/")

from core.evaluation.utils.sascorer import compute_sa_score
import argparse

pdb_blocks = {}
pdb_proteins = {}
molist = []
show_global = False
pc = PoseCheck()

def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper

def load_results_from_pt(pt_path):
    with open(pt_path, 'rb') as f:
        metrics = torch.load(f)
        if 'all_results' in metrics: # TargetDiff style
            results = metrics['all_results']
        else: # Our style
            results = []
            for graph in metrics:
                if 'mol' not in graph: continue
                data = {'mol': graph.mol, 'ligand_filename': graph.ligand_filename}
                results.append(data)
    return results

def load_mols_from_pt(pt_path):
    # results = load_results_from_pt(pt_path)
    results = torch.load(pt_path)
    molist = []
    for idx, r in enumerate(results):
        try:
            mol = r['mol']
            mol.SetProp('_Name', r['ligand_filename'])
            mol.SetProp('vina_score', str(r['vina']['score_only'][0]['affinity']))
            mol.SetProp('vina_minimize', str(r['vina']['minimize'][0]['affinity']))
            if 'dock' in r['vina']:
                mol.SetProp('vina_dock', str(r['vina']['dock'][0]['affinity']))
            mol.SetProp('strain', str(r['pose_check']['strain']))
        except Exception as e:
            print(e, idx)
            break
        molist.append(mol)
    return molist

def get_hbond(mol, protein_root):
                return None



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol_dir', type=str)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    args = parser.parse_args()

    mol_dirs = [args.mol_dir]
    ligand_fn = None
    for mol_dir in mol_dirs:
        mol_fns = glob.glob(f"{mol_dir.rstrip('/')}/*.sdf")
        if 'diff' in mol_dir or 'flag' in mol_dir:
            mol_fns = sorted(mol_fns, key=lambda x: int(os.path.basename(x).split('.')[0]))
        else:
            mol_fns = sorted(mol_fns, key=lambda x: int(os.path.basename(x).split('.')[0]) % 100)
        vina_scores = []
        vina_mins = []
        vina_docks = []
        clashes = []
        strain_energies = []
        qeds = []
        sas = []
        sizes = []
        hbond_acceptors = []
        hbond_donors = []

        # latest_time = os.path.getctime(f'{mol_dir}/968.sdf')
        for mol_fn in tqdm(mol_fns, total=len(mol_fns), desc=os.path.basename(mol_dir)):
            os.makedirs(mol_dir + '_out', exist_ok=True)

            # use sdf supplier
            suppl = Chem.SDMolSupplier(mol_fn, removeHs=False)
            mol = suppl[0]
            molist.append(mol)

            idx = int(os.path.basename(mol_fn).split('.')[0])

            if os.path.exists(f'{mol_dir}_out/{idx}.sdf'): continue
            if mol is None: continue

            try:
                if not mol.HasProp('vina_score'): continue
                smiles = Chem.MolToSmiles(mol)
                if '.' in smiles: continue
                # if mol is None or not mol.HasProp('vina_score'): continue
                
                # record vina affinities
                vina_scores.append(float(mol.GetProp('vina_score')))
                if mol.HasProp('vina_minimize'):
                    vina_mins.append(float(mol.GetProp('vina_minimize')))
                elif mol.HasProp('vina_min'):
                    vina_mins.append(float(mol.GetProp('vina_min')))
                if mol.HasProp('vina_dock'): vina_docks.append(float(mol.GetProp('vina_dock')))
                
                # calculate & record clash and strain
                if mol.GetProp('_Name') != ligand_fn:
                    ligand_fn = mol.GetProp('_Name')
                    protein_fn = os.path.join(
                        args.protein_root,
                        os.path.dirname(ligand_fn),
                        os.path.basename(ligand_fn)[:10] + '.pdb'
                    )
                    pc.load_protein_from_pdb(protein_fn)
                pc.load_ligands_from_mols([mol])
                clash = pc.calculate_clashes()[0]
                strain = pc.calculate_strain_energy()[0]
                clashes.append(clash)
                if strain != strain:
                    strain = 1e10
                strain_energies.append(strain)

                qed_val = qed(mol)
                sa_score = compute_sa_score(mol)
                qeds.append(qed_val)
                sas.append(sa_score)
                sizes.append(mol.GetNumAtoms())

                mol.SetProp('clash', str(clash))
                mol.SetProp('strain', str(strain))

                # hb = get_hbond(mol, args.protein_root)
                # if hb is None: continue
                # (hb_acceptor_count, hb_donor_count) = hb
                # mol.SetProp('hb_acceptor', str(hb_acceptor_count))
                # mol.SetProp('hb_donor', str(hb_donor_count))
                # hbond_acceptors.append(hb_acceptor_count)
                # hbond_donors.append(hb_donor_count)

                with Chem.SDWriter(f'{mol_dir}_out/{idx}.sdf') as w:
                    w.write(mol)

            except Exception as e:
                print(e, idx)
                continue

        print('file', mol_dir)
        if len(vina_scores) == 0: continue
        print('vina_score', np.mean(vina_scores), np.median(vina_scores), np.std(vina_scores))
        print('vina_min', np.mean(vina_mins), np.median(vina_mins), np.std(vina_mins))
        if vina_docks: print('vina_dock', np.mean(vina_docks), np.median(vina_docks), np.std(vina_docks))
        if clashes: print('clash', np.mean(clashes), np.median(clashes), np.std(clashes))

        if strain_energies: print('strain', np.quantile(strain_energies, 0.25), np.median(strain_energies), np.quantile(strain_energies, 0.75), np.std(strain_energies))
        print('qed', np.mean(qeds), np.median(qeds), np.std(qeds))
        print('sa', np.mean(sas), np.median(sas), np.std(sas))
        print('size', np.mean(sizes), np.median(sizes), np.std(sizes))
        # print('hbond_acceptors', sum(hbond_acceptors) / len(hbond_acceptors))
        # print('hbond_donors', sum(hbond_donors) / len(hbond_donors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--mol_dir', type=str)
    parser.add_argument('--protein_root', type=str, default='/data/mntdata/aaa/molsolver/data/test_set')
    args = parser.parse_args()
    # mol_path = '/data/mntdata/aaa/testmolcraft/logs/annoy_bfn_sbdd/official/default/test_outputs_v124/20241028-152757/vina_docked.pt'
    mol_path = '/data/mntdata/aaa/molsolver/test/samples/steps80_docked_rmsd.pt'
    # mol_path = '/data/mntdata/aaa/molsolver/logs/annoy_bfn_sbdd/official/default/test_outputs_v325/20250221-095056/vina_docked.pt'
    molist = load_mols_from_pt(mol_path)
    results = torch.load(mol_path)
    # molist = results
    mol_dir = '/data/mntdata/aaa/molsolver/results_sde'
    idx = 0
    vina_scores = []
    vina_mins = []
    vina_docks = []
    clashes = []
    # strain_energies = []
    qeds = []
    sas = []
    sizes = []
    hbond_acceptors = []
    hbond_donors = []
    ligand_fn = None
    for res, mol in tqdm(zip(results, molist), total=len(results), desc=os.path.basename(mol_dir)):
        idx += 1
        os.makedirs(mol_dir + '_out', exist_ok=True)
        if not mol.HasProp('vina_score'): continue
        smiles = Chem.MolToSmiles(mol)
        if '.' in smiles: continue
        try:
        # vina_scores.append(float(mol.GetProp('vina_score')))
        # if mol.HasProp('vina_minimize'):
        #     vina_mins.append(float(mol.GetProp('vina_minimize')))
        # elif mol.HasProp('vina_min'):
        #     vina_mins.append(float(mol.GetProp('vina_min')))
        # if mol.HasProp('vina_dock'): vina_docks.append(float(mol.GetProp('vina_dock')))
        # calculate & record clash and strain
            if mol.GetProp('_Name') != ligand_fn:
                ligand_fn = mol.GetProp('_Name')
                protein_fn = os.path.join(
                    args.protein_root,
                    os.path.dirname(ligand_fn),
                    os.path.basename(ligand_fn)[:10] + '.pdb'
                )
                pc.load_protein_from_pdb(protein_fn)
            pc.load_ligands_from_mols([mol])
            clash = pc.calculate_clashes()[0]
            # print(clash)
            res['pose_check']['clash'] = clash
            # strain = pc.calculate_strain_energy()[0]
            clashes.append(clash)
            # if strain != strain:
            #     strain = 1e10
            # strain_energies.append(strain)

            # qed_val = qed(mol)
            # sa_score = compute_sa_score(mol)
            # qeds.append(qed_val)
            # sas.append(sa_score)
            # sizes.append(mol.GetNumAtoms())

            # mol.SetProp('clash', str(clash))
            # mol.SetProp('strain', str(strain))
        
            # with Chem.SDWriter(f'{mol_dir}_out/{idx}.sdf') as w:
            #     w.write(mol)
        except Exception as e:
            print(e, idx)
            continue
    torch.save(results, "steps80_docked_rmsd_clash.pt")
    print('file', mol_dir)
    print('vina_score', np.mean(vina_scores), np.median(vina_scores), np.std(vina_scores))
    print('vina_min', np.mean(vina_mins), np.median(vina_mins), np.std(vina_mins))
    if vina_docks: print('vina_dock', np.mean(vina_docks), np.median(vina_docks), np.std(vina_docks))
    if clashes: print('clash', np.mean(clashes), np.median(clashes), np.std(clashes))
    # if strain_energies: print('strain', np.quantile(strain_energies, 0.25), np.median(strain_energies), np.quantile(strain_energies, 0.75), np.std(strain_energies))
    print('qed', np.mean(qeds), np.median(qeds), np.std(qeds))
    print('sa', np.mean(sas), np.median(sas), np.std(sas))
    print('size', np.mean(sizes), np.median(sizes), np.std(sizes))
    # print('hbond_acceptors', sum(hbond_acceptors) / len(hbond_acceptors))
    # print('hbond_donors', sum(hbond_donors) / len(hbond_donors))