#!/usr/bin/env python
"""Load gene expressions from pickle file.

Args:
    #1: STAR count file.
    #2: A featureCount file.
    #3: Output directory or file name.
        If this argument ends with '.csv', all expression levels are
            going to be stored in that file.
        If this argument ends with something else, a directory with
            such name will be created and expression levels for each
            chromosome as CSV files as well as a pickle file for all
            the expression levels will be created in that directory.

Functions:
    main: Load, trim and normalize gene expression levels.
    load_exp: Load gene expression levels from pickle file.
    get_gene_length: Load gene lengths from a featureCounts file.
    
Module-level variables:
    excluded_list: A list of sample IDs to be excluded by default.
"""
import pandas as pd
import sys
import os
import normalize


# Select subset of data.
excluded_list = [
    'YH09_0A4',
    'YH16_1A10',
    'YH09_1A3',
    'YH31_1A4',
    'YH03_1A5',
    'YH39_1A6',
    'YH19_1A7',
    'YH27_1A8',
    '1A07',
    'YH10_1B6',
    'YH21_1C3',
    'YH16_1C4',
    'YH23_1C5',
    'YH07_1D12',
    '1D12',
    'YH15_1E10',
    'YH23_1E4',
    'YH40_1E5',
    'YH06_1E6',
    'YH28_1F4',
    'YH01_1G11',
    'YH39_1G5',
    'YH34_1H12',
    'YH30_1H5',
    'YH24_2A12',
    'YH06_2A8',
    'YH07_2B9',
    'YH36_2C1',
    'YH07_2E12',
    'YH12_2F9',
    'YH13_2G3',
    'YH24_2G6',
    'YH08_3A9',
    'YH09_3B9',
    'YH16_3C1',
    '3C01',
    'YH08_3C12',
    'YH20_3C3',
    'YH15_3D10',
    'YH03_3D4',
    'YH18_3D9',
    'YH03_3F1',
    'YH07_3F12',
    'YH12_3F2',
    'YH04_3F3',
    'YH18_3G10',
    'YH04_3G3',
    'YH05_3G4',
    'YH11_3G5',
    'YH12_3G8',
    'YH06_3G9',
    'YH04_3H4',
    '4F12',
    'YH07_4F2',
    'YH03_4F3',
    'YH08_5F8',
    'YH12_5G4',
    'YH06_6A5',
    'YH12_6B10',
    'YH03_6C11',
    '6C11',
    'YH20_6C2',
    'YH12_6C3',
    'YH06_6D3',
    '6D03',
    'YH08_6F4',
    'YH23_6G3',
    'YH23_6G5',
    'YH08_6G7',
    'YH23_6H8',
    'YH09_7B8',
    'YH06_7C4',
    'YH20_7C7',
    'YH19_7D12',
    'YH12_7E11',
    'YH22_7E4',
    'YH11_7G4',
    'YH20_7H11',
    'YH08_7H9',
    'YH20_8H6',
    '1D02']


def main(argv):
    star_count_file = argv[0]
    feature_count_file = argv[1]
    df_gene_exp = load_exp(star_count_file, feature_count_file)
    output = argv[2]
    if output[-4:] == '.csv':
        # Do single csv file output.
        df_gene_exp.to_csv(output)
    else:
        target_dir = output
        # Create directory.
        os.makedirs(target_dir)
        df_gene_exp.to_pickle(target_dir+'/tpm.pkl')
        # Output to files by chromosome.
        for i in range(20):
            chr_start = '{0:0=2d}'.format(i+1)
            chr_end = '{0:0=2d}'.format(i+2)
            gene_id_start = 'Glyma.{}G000000'.format(chr_start)
            gene_id_end = 'Glyma.{}G000000'.format(chr_end)
            df_gene_exp.loc[gene_id_start:gene_id_end].to_csv(
                target_dir+'/tpm-chr{}.csv'.format(chr_start)
                )
        df_gene_exp.loc['Glyma.21G000000':].to_csv(
            target_dir+'/tpm-scaffold.csv'
            )
    return


def load_exp(pkl_file, feature_count_file):
    """Load gene expression levels from a pickle file.
    
    Args:
        pkl_file: pickle file of the raw counts.
        feature_count_file: Any featureCount file.
        
    Returns:
        A DataFrame of the gene expression levels in TPM.
    """
    df_counts = load_and_drop(pkl_file, excluded_list)
    sr_gene_lengths = get_gene_length(feature_count_file)
    # Effective gene length (relative mapping efficiency).
    sr_eff_lengths = sr_gene_lengths-16
    # Normalization by effective gene lengths first.
    df_gene_exp = (df_counts.T/sr_eff_lengths).T
    # Then normalize by library size, i.e., the number of mapped reads.
    df_gene_exp = df_gene_exp/df_gene_exp.sum()*1e6
    return df_gene_exp


def get_gene_length(feature_count_file):
    """Get gene lengths from featureCounts output.

    Args:
        feature_count_file: Any featureCount file.
    """
    df_fc = normalize.load_data(
        feature_count_file, quant_method='feature_counts'
        )
    sr_gene_lengths = normalize.extract_column(df_fc, 'Length')
    return sr_gene_lengths


def load_and_drop(pkl_file, drop_list=excluded_list):
    """Load raw counts and drop duplicate samples.

    Args:
        pkl_file: pickle file for raw counts.
        drop_list: A list of duplicate sample IDs.

    Returns:
        A DataFrame of raw counts without duplicates.
    """
    df_counts = pd.read_pickle(pkl_file).drop(
        drop_list, axis=1, errors='ignore'
        )
    df_counts.sort_index(axis=1, inplace=True)
    return df_counts


if __name__ == "__main__":
    main(sys.argv[1:])
