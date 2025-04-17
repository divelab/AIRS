from fcd_torch import FCD as FCDMetric
from moses.metrics.metrics import SNNMetric, FragMetric, ScafMetric, internal_diversity, \
    fraction_passes_filters, weight, logP, SA, QED
from multiprocessing import Pool
from rdkit import Chem
import copy
import numpy as np
import time


# obtain smiles from rdmol
def get_smiles(rdmol):
    rdmol = copy.deepcopy(rdmol)
    try:
        Chem.SanitizeMol(rdmol)
    except ValueError:
        return None
    smiles = Chem.MolToSmiles(rdmol, isomericSmiles=False)
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles
    return None


def reconstruct_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def mapper(n_jobs):
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None, fcd_only=False):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    # smiles = mapper(pool)(get_smiles, mols)
    smiles = list(set(smiles) - {None})
    # re_mols = mapper(pool)(reconstruct_mol, smiles)
    
    valid_smiles = []
    re_mols = []
    for smile in smiles:
        mol = reconstruct_mol(smile)
        if mol == None:
            continue
        else:
            re_mols.append(mol)
            valid_smiles.append(smile)
    statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(valid_smiles)
    if not fcd_only:
        statistics['SNN'] = SNNMetric(**kwargs).precalc(re_mols)
        statistics['Frag'] = FragMetric(**kwargs).precalc(re_mols)
        statistics['Scaf'] = ScafMetric(**kwargs).precalc(re_mols)
    # for name, func in [('logP', logP), ('SA', SA),
    #                    ('QED', QED),
    #                    ('weight', weight)]:
    #     statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics


def get_moses_metrics(test_smiles, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None):
    # compute intermediate statistics for test rdmols
    ptest = compute_intermediate_statistics(test_smiles, n_jobs=n_jobs, device=device,
                                            batch_size=batch_size, pool=ptest_pool)

    def moses_metrics(gen_mols, pool=None):
        metrics = {}
        if pool is None:
            if n_jobs != 1:
                pool = Pool(n_jobs)
                close_pool = True
            else:
                pool = 1
        kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        gen_smiles = mapper(pool)(get_smiles, gen_mols)
        gen_smiles = list(set(gen_smiles) - {None})
        re_mols = mapper(pool)(reconstruct_mol, gen_smiles)
        metrics['FCD'] = FCDMetric(**kwargs_fcd)(gen=gen_smiles, pref=ptest['FCD'])
        metrics['SNN'] = SNNMetric(**kwargs)(gen=re_mols, pref=ptest['SNN'])
        metrics['Frag'] = FragMetric(**kwargs)(gen=re_mols, pref=ptest['Frag'])
        metrics['Scaf'] = ScafMetric(**kwargs)(gen=re_mols, pref=ptest['Scaf'])
        metrics['IntDiv'] = internal_diversity(re_mols, pool, device=device)
        metrics['Filters'] = fraction_passes_filters(re_mols, pool)

        # drug properties
        metrics['QED'] = MeanProperty(re_mols, QED, n_jobs)
        metrics['SA'] = MeanProperty(re_mols, SA, n_jobs)
        metrics['logP'] = MeanProperty(re_mols, logP, n_jobs)
        metrics['weight'] = MeanProperty(re_mols, weight, n_jobs)

        if close_pool:
            pool.close()
            pool.join()
        return metrics

    return moses_metrics


def get_fcd_metric(test_mols, n_jobs=1, device='cpu', batch_size=2000, ptest_pool=None):
    ptest = compute_intermediate_statistics(test_mols, n_jobs=n_jobs, device=device,
                                            batch_size=batch_size, pool=ptest_pool, fcd_only=True)

    def fcd_metric(gen_mols, pool=None):
        metrics = {}
        if pool is None:
            if n_jobs != 1:
                pool = Pool(n_jobs)
                close_pool = True
            else:
                pool = 1
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        gen_smiles = mapper(pool)(get_smiles, gen_mols)
        gen_smiles = list(set(gen_smiles) - {None})
        try:
            metrics['FCD'] = FCDMetric(**kwargs_fcd)(gen=gen_smiles, pref=ptest['FCD'])
        except:
            metrics['FCD'] = float('nan')
        if close_pool:
            pool.close()
            pool.join()
        return metrics
    return fcd_metric


def MeanProperty(mols, func, n_jobs=1):
    values = mapper(n_jobs)(func, mols)
    return np.mean(np.array(values))
