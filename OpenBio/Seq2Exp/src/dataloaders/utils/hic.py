import numpy as np
import scipy.sparse as ssp
import pandas as pd
import time, os
import json
import h5py
import hicstraw
import csv
from tqdm import tqdm


def get_hic_file(chromosome, hic_dir, allow_vc=True, hic_type="juicebox"):
    if hic_type == "juicebox":
        hic_file = os.path.join(hic_dir, chromosome, chromosome + ".KRobserved.gz")
        hic_norm = os.path.join(hic_dir, chromosome, chromosome + ".KRnorm.gz")

        is_vc = False
        if allow_vc and not hic_exists(hic_file):
            hic_file = os.path.join(hic_dir, chromosome, chromosome + ".VCobserved.gz")
            hic_norm = os.path.join(hic_dir, chromosome, chromosome + ".VCnorm.gz")

            if not hic_exists(hic_file):
                RuntimeError("Could not find KR or VC normalized hic files")
            else:
                print("Could not find KR normalized hic file. Using VC normalized hic file")
                is_vc = True

        print("Using: " + hic_file)
        return hic_file, hic_norm, is_vc
    elif hic_type == "bedpe":
        hic_file = os.path.join(hic_dir, chromosome, chromosome + ".bedpe.gz")

        return hic_file, None, None


def hic_exists(file):
    if not os.path.exists(file):
        return False
    elif file.endswith('gz'):
        # gzip file still have some size. This is a hack
        return (os.path.getsize(file) > 100)
    else:
        return (os.path.getsize(file) > 0)


def load_hic(hic_file, hic_norm_file, hic_is_vc, hic_type, hic_resolution, tss_hic_contribution, window, min_window,
             gamma, interpolate_nan=True, apply_diagonal_bin_correction=True):
    print("Loading HiC")

    if hic_type == 'juicebox':
        HiC_sparse_mat = hic_to_sparse(hic_file, hic_norm_file, hic_resolution)
        HiC = process_hic(hic_mat=HiC_sparse_mat,
                          hic_norm_file=hic_norm_file,
                          hic_is_vc=hic_is_vc,
                          resolution=hic_resolution,
                          tss_hic_contribution=tss_hic_contribution,
                          window=window,
                          min_window=min_window,
                          gamma=gamma,
                          interpolate_nan=interpolate_nan,
                          apply_diagonal_bin_correction=apply_diagonal_bin_correction)
        # HiC = juicebox_to_bedpe(HiC, chromosome, args)
    elif hic_type == 'bedpe':
        HiC = pd.read_csv(hic_file, sep="\t", names=['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2', 'name', 'hic_contact'])

    return HiC


def process_hic(hic_mat, hic_norm_file, hic_is_vc, resolution, tss_hic_contribution, window, min_window=0,
                hic_is_doubly_stochastic=False, apply_diagonal_bin_correction=True, interpolate_nan=True, gamma=None,
                kr_cutoff=.25):
    # Make doubly stochastic.
    # Juicer produces a matrix with constant row/column sums. But sum is not 1 and is variable across chromosomes
    t = time.time()

    if not hic_is_doubly_stochastic and not hic_is_vc:
        # Any row with Nan in it will sum to nan
        # So need to calculate sum excluding nan
        temp = hic_mat
        temp.data = np.nan_to_num(temp.data, copy=False)
        sums = temp.sum(axis=0)
        sums = sums[~np.isnan(sums)]
        assert (np.max(sums[sums > 0]) / np.min(sums[sums > 0]) < 1.001)
        mean_sum = np.mean(sums[sums > 0])

        if abs(mean_sum - 1) < .001:
            print('HiC Matrix has row sums of {}, continuing without making doubly stochastic'.format(mean_sum))
        else:
            print('HiC Matrix has row sums of {}, making doubly stochastic...'.format(mean_sum))
            hic_mat = hic_mat.multiply(1 / mean_sum)

    # Adjust diagonal of matrix based on neighboring bins
    # First and last rows need to be treated differently
    if apply_diagonal_bin_correction:
        last_idx = hic_mat.shape[0] - 1
        nonzero_diag = hic_mat.nonzero()[0][hic_mat.nonzero()[0] == hic_mat.nonzero()[1]]
        nonzero_diag = list(set(nonzero_diag) - set(np.array([last_idx])) - set(np.array([0])))

        for ii in nonzero_diag:
            hic_mat[ii, ii] = max(hic_mat[ii, ii - 1], hic_mat[ii, ii + 1]) * tss_hic_contribution / 100

        if hic_mat[0, 0] != 0:
            hic_mat[0, 0] = hic_mat[0, 1] * tss_hic_contribution / 100

        if hic_mat[last_idx, last_idx] != 0:
            hic_mat[last_idx, last_idx] = hic_mat[last_idx, last_idx - 1] * tss_hic_contribution / 100

    # Any entries with low KR norm entries get set to NaN. These will be interpolated below
    hic_mat = apply_kr_threshold(hic_mat, hic_norm_file, kr_cutoff)

    # Remove lower triangle
    if not hic_is_vc:
        hic_mat = ssp.triu(hic_mat)
    else:
        hic_mat = process_vc(hic_mat)

    # Turn into dataframe
    hic_mat = hic_mat.tocoo(copy=False)
    hic_df = pd.DataFrame({'bin1': hic_mat.row, 'bin2': hic_mat.col, 'hic_contact': hic_mat.data})

    # Prune to window
    hic_df = hic_df.loc[np.logical_and(abs(hic_df['bin1'] - hic_df['bin2']) <= window / resolution,
                                       abs(hic_df['bin1'] - hic_df['bin2']) >= min_window / resolution)]
    print("HiC has {} rows after windowing between {} and {}".format(hic_df.shape[0], min_window, window))

    # Fill NaN
    # NaN in the KR normalized matrix are not zeros. They are entries where the KR algorithm did not converge (or low KR norm)
    # So need to fill these. Use powerlaw.
    # Not ideal obviously but the scipy interpolation algos are either very slow or don't work since the nan structure implies that not all nans are interpolated
    if interpolate_nan:
        nan_loc = np.isnan(hic_df['hic_contact'])
        hic_df.loc[nan_loc, 'hic_contact'] = get_powerlaw_at_distance(
            abs(hic_df.loc[nan_loc, 'bin1'] - hic_df.loc[nan_loc, 'bin2']) * resolution, gamma)

    print('process.hic: Elapsed time: {}'.format(time.time() - t))

    return (hic_df)


def apply_kr_threshold(hic_mat, hic_norm_file, kr_cutoff):
    # Convert all entries in the hic matrix corresponding to low kr norm entries to NaN
    # Note that in scipy sparse matrix multiplication 0*nan = 0
    # So this doesn't convert 0's to nan only nonzero to nan

    norms = np.loadtxt(hic_norm_file)
    norms[norms < kr_cutoff] = np.nan
    norms[norms >= kr_cutoff] = 1
    norm_mat = ssp.dia_matrix((1.0 / norms, [0]), (len(norms), len(norms)))

    return norm_mat * hic_mat * norm_mat


def hic_to_sparse(filename, norm_file, resolution, hic_is_doubly_stochastic=False):
    t = time.time()
    HiC = pd.read_table(filename, names=["bin1", "bin2", "hic_contact"],
                        header=None, engine='c', memory_map=True)

    # verify our assumptions
    assert np.all(HiC.bin1 <= HiC.bin2)

    # Need load norms here to know the dimensions of the hic matrix
    norms = pd.read_csv(norm_file, header=None)
    hic_size = norms.shape[0]

    # convert to sparse matrix in CSR (compressed sparse row) format, chopping
    # down to HiC bin size.  note that conversion to scipy sparse matrices
    # accumulates repeated indices, so this will do the right thing.
    row = np.floor(HiC.bin1.values / resolution).astype(int)
    col = np.floor(HiC.bin2.values / resolution).astype(int)
    dat = HiC.hic_contact.values

    # JN: Need both triangles in order to compute row/column sums to make double stochastic.
    # If juicebox is upgraded to return DS matrices, then can remove one triangle
    # TO DO: Remove one triangle when juicebox is updated.
    # we want a symmetric matrix.  Easiest to do that during creation, but have to be careful of diagonal
    if not hic_is_doubly_stochastic:
        mask = (row != col)  # off-diagonal
        row2 = col[mask]  # note the row/col swap
        col2 = row[mask]
        dat2 = dat[mask]

        # concat and create
        row = np.hstack((row, row2))
        col = np.hstack((col, col2))
        dat = np.hstack((dat, dat2))

    print('hic.to.sparse: Elapsed time: {}'.format(time.time() - t))

    return ssp.csr_matrix((dat, (row, col)), (hic_size, hic_size))


def get_powerlaw_at_distance(distances, gamma, min_distance=5000, scale=None):
    assert (gamma > 0)

    # The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    # So just assume that everything at < 5kb is equal to 5kb.
    # TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    # Determine scale parameter
    # A powerlaw distribution has two parameters: the exponent and the minimum domain value
    # In our case, the minimum domain value is always constant (equal to 1 HiC bin) so there should only be 1 parameter
    # The current fitting approach does a linear regression in log-log space which produces both a slope (gamma) and a intercept (scale)
    # Empirically there is a linear relationship between these parameters (which makes sense since we expect only a single parameter distribution)
    # It should be possible to analytically solve for scale using gamma. But this doesn't quite work since the hic data does not actually follow a power-law
    # So could pass in the scale parameter explicity here. Or just kludge it as I'm doing now
    # TO DO: Eventually the pseudocount should be replaced with a more appropriate smoothing procedure.

    # 4.80 and 11.63 come from a linear regression of scale on gamma across 20 hic cell types at 5kb resolution. Do the params change across resolutions?
    if scale is None:
        scale = -4.80 + 11.63 * gamma

    powerlaw_contact = np.exp(scale + -1 * gamma * log_dists)

    return (powerlaw_contact)


def process_vc(hic):
    # For a vc normalized matrix, need to make rows sum to 1.
    # Assume rows correspond to genes and cols to enhancers

    row_sums = hic.sum(axis=0)
    row_sums[row_sums == 0] = 1
    norm_mat = ssp.dia_matrix((1.0 / row_sums, [0]), (row_sums.shape[1], row_sums.shape[1]))

    # left multiply to operate on rows
    hic = norm_mat * hic

    return (hic)


def process_hic_signals(cell_type, total_len=200_000, genome='hg38', root_path=''):
    each_total_lens = total_len // 2
    with open(os.path.join(root_path, 'chr_length.json'), 'r') as file:
        chr_lengths = json.load(file)

    root_data_folder = os.path.join(root_path, 'promo_enhan_inter')
    all_chrs = ['chr' + str(i) for i in list(range(1, 23)) + ['X']]

    if cell_type == 'K562':
        promoter_df = pd.read_csv(os.path.join(root_data_folder,
                                               'K562_DNase_ENCFF257HEE_hic_4DNFITUOMFUQ_1MB_ABC_nominated/DNase_ENCFF257HEE_Neighborhoods/GeneList.txt'),
                                  sep='\t', index_col='symbol')
    elif cell_type == 'GM12878':
        promoter_df = pd.read_csv(os.path.join(root_data_folder,
                                               'GM12878_DNase_ENCFF020WZB_hic_4DNFI1UEG1HD_1MB_ABC_nominated/DNase_ENCFF020WZB_Neighborhoods/GeneList.txt'),
                                  sep='\t', index_col='symbol')
    elif cell_type == 'H1':
        promoter_df = pd.read_csv(os.path.join(root_data_folder, 'H1_ABC/Neighborhoods/GeneList.txt'),
                                  sep='\t', index_col='symbol')

    promoter_df = promoter_df[['chr', 'tss']]

    for each_chr in all_chrs:
        print(f"current {each_chr}")
        hic_feat_dict = {}
        chrom_length = chr_lengths[genome][each_chr]

        csv_file = os.path.join(root_data_folder, 'HiC', cell_type, f'hic_contacts_{each_chr}.csv')
        HiC = pd.read_csv(csv_file)

        cur_promoter_df = promoter_df[promoter_df['chr'] == each_chr]
        unique_tss = cur_promoter_df['tss'].unique()
        for each_tss in tqdm(unique_tss):
            start = max(0, each_tss - each_total_lens)
            end = min(chrom_length, each_tss + each_total_lens)
            left_padding = max(0, each_total_lens - each_tss)
            right_padding = max(0, (each_tss + each_total_lens) - chrom_length)

            # build new df
            df_pro_bin = pd.DataFrame({'chr': [each_chr] * (end - start)})
            df_pro_bin['start'] = list(range(start, end))
            df_pro_bin['end'] = list(range(start, end))
            df_pro_bin['TargetGeneTSS'] = [each_tss] * (end - start)
            pred = make_pred_table(df_pro_bin)
            pred = annotate_predictions(pred)
            pred = add_powerlaw_to_predictions(pred)

            pred['enh_bin'] = np.floor(pred['enh_midpoint'] / 5000).astype(int)
            pred['tss_bin'] = np.floor(pred['TargetGeneTSS'] / 5000).astype(int)
            pred['bin1'] = np.amin(pred[['enh_bin', 'tss_bin']], axis = 1)
            pred['bin2'] = np.amax(pred[['enh_bin', 'tss_bin']], axis = 1)
            pred = pred.merge(HiC, how = 'left', on = ['bin1','bin2'])
            pred.fillna(value={'hic_contact' : 0}, inplace=True)

            # QC juicebox HiC
            pred = qc_hic(pred)

            pred.drop(['bin1', 'bin2', 'enh_idx', 'enh_midpoint', 'tss_bin', 'enh_bin'], inplace=True, axis=1, errors='ignore')

            # Add powerlaw scaling
            pred = scale_hic_with_powerlaw(pred)

            # Add pseudocount
            pred = add_hic_pseudocount(pred)

            # hic_contact_pl_scaled_adj
            # powerlaw_contact_reference
            feat = pred['hic_contact_pl_scaled_adj'].to_list()
            feat = [0.0] * left_padding + feat + [0.0] * right_padding

            hic_feat_dict[f'{each_chr}_{each_tss}'] = feat

        with h5py.File(os.path.join(root_data_folder, 'HiC', f'{cell_type}', f'hic_feat_{each_chr}.h5'), 'w') as f:
            for key, features in hic_feat_dict.items():
                f.create_dataset(key, data=features)


def make_pred_table(enh):
    enh['enh_midpoint'] = (enh['start'] + enh['end']) / 2
    enh['enh_idx'] = enh.index
    # genes['gene_idx'] = genes.index
    enh['Chromosome'] = enh['chr']
    # enh_pr = df_to_pyranges(enh)
    # genes_pr = df_to_pyranges(genes, start_col='TargetGeneTSS', end_col='TargetGeneTSS', start_slop=args.window,
    #                           end_slop=args.window)
    #
    # pred = enh_pr.join(genes_pr).df.drop(['Start_b', 'End_b', 'chr_b', 'Chromosome', 'Start', 'End'], axis=1)
    pred = enh
    pred['distance'] = abs(pred['enh_midpoint'] - pred['TargetGeneTSS'])
    pred = pred.loc[pred['distance'] < 5000000, :]  # for backwards compatability

    return pred


def annotate_predictions(pred, tss_slop=500):
    #TO DO: Add is self genic
    pred['isSelfPromoter'] = np.logical_and.reduce((pred.start - tss_slop < pred.TargetGeneTSS,
                                                    pred.end + tss_slop > pred.TargetGeneTSS))

    return(pred)


def add_powerlaw_to_predictions(pred):
    pred['powerlaw_contact'] = get_powerlaw_at_distance(pred['distance'].values, 0.87)
    pred['powerlaw_contact_reference'] = get_powerlaw_at_distance(pred['distance'].values, 0.87)
    return pred


def qc_hic(pred, threshold = .01):
    # Genes with insufficient hic coverage should get nan'd

    summ = pred.loc[pred['isSelfPromoter'],:].groupby(['TargetGeneTSS']).agg({'hic_contact' : 'sum'})
    bad_genes = summ.loc[summ['hic_contact'] < threshold,:].index

    pred.loc[pred['TargetGeneTSS'].isin(bad_genes), 'hic_contact'] = np.nan

    return pred


def scale_hic_with_powerlaw(pred):
    #Scale hic values to reference powerlaw
    pred['hic_contact_pl_scaled'] = pred['hic_contact'] * (pred['powerlaw_contact_reference'] / pred['powerlaw_contact'])
    return(pred)


def add_hic_pseudocount(pred):
    # Add a pseudocount based on the powerlaw expected count at a given distance
    powerlaw_fit = get_powerlaw_at_distance(pred['distance'].values, 0.87)
    powerlaw_fit_at_ref = get_powerlaw_at_distance(1000000.0, 0.87)

    pseudocount = np.amin(pd.DataFrame({'a': powerlaw_fit, 'b': powerlaw_fit_at_ref}), axis=1)
    pred['hic_pseudocount'] = pseudocount
    pred['hic_contact_pl_scaled_adj'] = pred['hic_contact_pl_scaled'] + pseudocount

    return (pred)


def hic_pre_processing(cell_type, folder):
    # read hic file
    if cell_type == 'K562':
        hic_file = os.path.join(folder, cell_type, '4DNFITUOMFUQ.hic')
    elif cell_type == 'GM12878':
        hic_file = os.path.join(folder, cell_type, '4DNFI1UEG1HD.hic')
    elif cell_type == 'H1':
        hic_file = os.path.join(folder, cell_type, '4DNFIFI6NIKJ.hic')

    hic = hicstraw.HiCFile(hic_file)
    chromosomes = hic.getChromosomes()
    print(f'process cell type {cell_type}')
    for chrom in chromosomes:
        print(f"Name: {chrom.name}, Length: {chrom.length}")

    all_chr = [str(i) for i in list(range(1, 23)) + ['X']]

    for chrm in all_chr:
        chromosome1 = str(chrm)
        chromosome2 = str(chrm)
        data_type = 'observed'
        normalization = 'VC'  # NONE, KR
        resolution = 5000
        unit = 'BP'
        gr1 = 0
        # gr2 = 1000000
        gr2 = next((chrom for chrom in chromosomes if chrom.name == chromosome1), None).length  # chr1 的结束位置
        gc1 = 0
        # gc2 = 1000000
        gc2 = next((chrom for chrom in chromosomes if chrom.name == chromosome2), None).length  # chr1 的结束位置

        mzd = hic.getMatrixZoomData(chromosome1, chromosome2, data_type, normalization, unit, resolution)
        # numpy_matrix = mzd.getRecordsAsMatrix(gr1, gr2, gc1, gc2)
        records = mzd.getRecords(gr1, gr2, gc1, gc2)
        # print(f"binX: {records[0].binX}, binY: {records[0].binY}, counts: {records[0].counts}")

        print(f"chr {chrm}, length {len(records)}")

        save_file = os.path.join(folder, cell_type, f'hic_contacts_chr{chrm}.csv')
        with open(save_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入CSV文件的头部
            writer.writerow(['bin1', 'bin2', 'hic_contact'])

            for record in records:
                bin1 = record.binX // resolution
                bin2 = record.binY // resolution
                hic_contact = record.counts
                writer.writerow([bin1, bin2, hic_contact])


if __name__ == "__main__":
    dataset_root_path = ""
    HiC_path = os.path.join(dataset_root_path, 'promo_enhan_inter', 'HiC')
    cell_type = 'H1'
    # hic_pre_processing(cell_type, HiC_path)
    process_hic_signals(cell_type, root_path=dataset_root_path)
