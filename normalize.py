#!/usr/bin/env python
"""Get NFPKM from STAR counts."""
import pandas as pd
import numpy as np
import re
import sys
import os
import fnmatch


def main(argv):
    star_count_dir = '/home/xiaohank/scratch/star-runs/raw-alignment/'
    feature_count_file = '/scratch/users/xiaohank/job-runs/feature_count-5780638/YH01_0E4_GAGATTCC-AGGCGAAG_L001_R1_001_featCounts.txt'
    mean_fragment_length = 300
    df_fc = load_data(feature_count_file, quant_method='feature_counts')
    sr_gene_lengths = extract_column(df_fc, 'Length')
    sr_eff_lengths = get_eff_lengths(sr_gene_lengths, mean_fragment_length)
    star_files = find_star_files(star_count_dir)
    sample_ids = get_sample_ids(star_files)
    df_nfpkm = pd.DataFrame()
    for idx, single_star_file in enumerate(star_files):
        sr_nfpkm = get_nfpkm_from_star_counts(single_star_file, sr_eff_lengths)
        sr_nfpkm.name = sample_ids[idx]
        df_nfpkm = pd.concat([df_nfpkm, sr_nfpkm], axis=1)
    df_nfpkm.to_pickle(argv[0])
    return None


def find_star_files(dir):
    """Find files with suffix 'ReadsPerGene.out.tab' recursively."""
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, '*ReadsPerGene.out.tab'):
            matches.append(os.path.join(root, filename))
    return matches


def get_sample_ids(filenames):
    """DEPRECATED: Get sample IDs from sample filenames."""
    return [re.search(r'^.*/([A-Z0-9]+_[A-Z0-9]+)', x).group(1) for x in filenames]


def get_sample_id(filename):
    """Get the sample ID from the sample filename.
    
    The sample ID is after the last backslash, and before an underscore
    followed by the index sequence of eight nucleotides (A, T, C or G).
    
    Args:
        filename: The name of the STAR ReadsPerGene.out.tab file.

    Returns:
        Sample ID, or empty string if no indices are found.
    """
    filename_no_dir = filename.split('/')[-1]
    try:
        sample_id = re.search(r'(.*)_[ATCG]{8}', filename_no_dir).group(1)
    except AttributeError:
        # No 8-nt indices found.
        sample_id = ''
    return sample_id


def get_nfpkm_from_star_counts(star_file, sr_eff_lengths):
    """Get NFPKM quantification from STAR counts.

    Args:
        star_file: STAR count file with suffix 'ReadsPerGene.out.tab'.
        sr_eff_lengths: A Series of effective lengths.

    Returns:
        A Series of NFPKM.
    """
    df_star = load_data(star_file, quant_method='star')
    sr_star_counts = extract_counts(df_star, quant_method='star')
    sr_nfpkm = get_nfpkm(sr_star_counts, sr_eff_lengths)
    return sr_nfpkm


def load_data(data_file, quant_method='star', gene_id_str='gene ID',
              unstrand_str='unstranded', fwstrand_str='forward stranded',
              rvstrand_str='reverse stranded'):
    """Load data from output of STAR, Kallisto or featureCounts.

    Args:
        data_file: Target file produced by the counting software.
        quant_method: Counting software; can be one of the following.
            * 'star'
            * 'kallisto'
            * 'feature_counts'
            
    Returns:
        A pandas DataFrame.
        
        The index is the default 0-based integers. The names of the columns are either extract from the data file
        (in the cases of Kallisto and featureCounts) or specified as module-level variables (in the case of STAR).
    """
    if quant_method == 'star':
        df = pd.read_csv(data_file, sep='\t', names=[gene_id_str, unstrand_str, fwstrand_str, rvstrand_str],
                         skiprows=4).sort_values(by=gene_id_str)
    elif quant_method == 'kallisto':
        df = pd.read_csv(data_file, sep='\t').sort_values(by='target_id').reset_index(drop=True)
    elif quant_method == 'feature_counts':
        df = pd.read_csv(data_file, sep='\t', skiprows=1).sort_values(by='Geneid')
    else:
        print('Unknown quantification method.')
        exit(1)
    return df


def extract_column(df, col_name='Length', icol=0):
    """Extract a column from a DataFrame and trim the gene IDs.
    
    Args:
        df: A DataFrame with long gene IDs in the first column.
        col_name (optional): The name of the column to be extract; default is 'Length'.
        icol (optional): The 0-based index for the column; overrides col_name if set to a positive integer.
    
    Returns:
        A Series of extracted column with shortened gene IDs as index.
    """
    # We trim the following suffix of the gene IDs to make them shorter in the returned Series.
    size_gene_suffix = len('.Wm82.a2.v1')
    gene_ids = pd.Series([gene[:-size_gene_suffix] for gene in df.iloc[:, 0]])
    if icol:
        sr_col = df.iloc[:, icol]
    else:
        sr_col = df[col_name]
    sr_col.index = gene_ids
    return sr_col


def extract_counts(df, quant_method='star', rvstrand_str='reverse stranded'):
    """Extract counts from the STAR, featureCounts or Kallisto DataFrame.
    
    Args:
        df: A DataFrame generated by load_data() from the STAR data file.
        quant_method: Counting software; can be one of the following.
            * 'star'
            * 'kallisto'
            * 'feature_counts'
        
    Returns:
        A Series with 'counts' as the name, read counts as the values, and the shortened gene IDs as the index.
    """
    if quant_method == 'star':
        sr_count_column = extract_column(df, rvstrand_str)
    elif quant_method == 'feature_counts':
        sr_count_column = extract_column(df, icol=6)
    elif quant_method == 'kallisto':
        # For Kallisto we need to combine the counts for different transcripts.
        # One naive way is simply adding the counts up without considering the lengths of the transcripts.
        sr_count_column = extract_column_per_gene(df, 3)
    else:
        print('Unknown quantification method.')
        exit(1)
    sr_count_column.rename('counts')
    return sr_count_column


def eff_length(gene_length, fragment_length):
    """Returns effective gene length."""
    x = gene_length-fragment_length
    if x > 0:
        return x
    else:
        return np.nan


def get_nfpkm(sr_counts, sr_eff_lengths):
    """Get normalized FPKM counts from raw counts.
    
    Args:
        sr_counts: A Series of raw counts.
        sr_eff_lengths: A Series of effective gene lengths.
        
    Returns:
        A Series of normalized FPKM.
    """
    sr_fpkm = sr_counts/sr_eff_lengths
    sr_nfpkm = sr_fpkm/sr_fpkm.sum()*1e6
    sr_nfpkm.name = 'nfpkm'
    return sr_nfpkm


def get_eff_lengths(sr_lengths, mean_fragment_length):
    """Get effective gene lengths from a Series of gene lengths.
    
    Args:
        sr_lengths: A Series of gene lengths.
        mean_fragment_length: The mean fragment length of the cDNA.
    
    Returns:
        A Series of effective gene lengths.
    """
    return pd.Series([eff_length(x, mean_fragment_length) for x in sr_lengths], index=sr_lengths.index,
                     name=sr_lengths.name)


if __name__ == "__main__":
    main(sys.argv[1:])

