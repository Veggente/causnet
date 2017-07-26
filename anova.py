#!/usr/bin/env python
"""ANOVA for differential expression analysis.

Args:
    #0: Pickle file for all the raw counts.
    #1: Faqiang's slides for sample ID parsing.
    #2: Faqiang's text file for sample ID parsing.
    #3: Variance factor selector.
        E.g., "1 2 4" for temperature + photoperiod + sample time.
    #4: Partition factor selector.
    #5: Output file name.
    #6 (for DEA only): Number of genes for ANOVA.
        -1 means all of the genes.
    #7 (for DEA only): featureCounts file.

Returns:
    Saves p-values in a pickle file.
"""
import pandas as pd
import re
import sys
import scipy.stats
import pickle
import os

import normalize
import load_exp


def main(argv):
    star_count_file = argv[0]
    faqiang_slides = argv[1]
    faqiang_txt = argv[2]
    var_factor = [int(x) for x in argv[3].split()]
    par_factor = [int(x) for x in argv[4].split()]
    result_path = argv[5]
    if len(argv) == 6:
        result = anova_lib_size(
            star_count_file, faqiang_slides, faqiang_txt, var_factor,
            par_factor
            )
    elif len(argv) == 8:
        num_genes_requested = int(argv[6])
        feature_count_file = argv[7]
        result = anova_tpm(
            star_count_file, feature_count_file, faqiang_slides,
            faqiang_txt, num_genes_requested, var_factor, par_factor
            )
    else:
        print('Invalid number of arguments.')
        exit(1)
    if result_path[-4:] == '.pkl':
        pickle.dump(result, open(result_path, 'wb'))
    else:
        if len(argv) == 6:
            with open(result_path, 'w') as f:
                for key in result:
                    value = result[key]
                    f.write(anova_header(key, par_factor, var_factor))
                    f.write('DOF between: {}\nDOF within: {}\n'.format(
                        value[1], value[2]
                        ))
                    f.write('Explained variance: {}\n'.format(value[3]))
                    f.write('p-value: {}\n\n'.format(value[0]))
        else:
            print('Need a pickle file for output of DEA.')
            exit(1)
    return None


# A map converting single-letter times to multi-letter times.
sample_time_converter = {
    'A': 'I1',
    'B': 'D0',
    'C': 'I3',
    'D': 'I4',
    'E': 'I5',
    'F': 'I6',
    'G': 'D1',
    'H': 'D2',
    'I': 'D3',
    'J': 'D4',
    'K': 'II1',
    'L': 'II2',
    'M': 'II3',
    'N': 'II4',
    'O': 'II5',
    'P': 'II6',
    'Q': 'D6',
    'R': 'D7'
}

lane_41_sample_ids = [
    '1_00E_06',
    '1_00E_10',
    '1F04',
    '1G11',
    '1H05',
    '1H12',
    '2_00E_12',
    '2G06',
    '3B09',
    '3C03',
    '3C12',
    '3D09',
    '3D10',
    '3F02',
    '3F12',
    '3G05',
    '3H04',
    '5F08',
    '6G03',
    '7_00E_04',
    '7C04',
    '7C07',
    '7D12',
    '7G04',
    '7H09']


lane_42_sample_ids = [
    '1_E04',
    '1_E05',
    '1A04',
    '1A10',
    '1C03',
    '1D02',
    '1D12',
    '2A08',
    '2B09',
    '2C01',
    '2F09',
    '2G03',
    '3D04',
    '3F01',
    '3G10',
    '4F02',
    '4F03',
    '4F12',
    '5G04',
    '6A05',
    '6D03',
    '6G05',
    '6G07',
    '7_E11',
    '7B08',
    '8H06']


lane_43_sample_ids = [
    '1A03',
    '1A05',
    '1A06',
    '1A07',
    '1A08',
    '1B06',
    '1C04',
    '1C05',
    '1G05',
    '2A12',
    '3A09',
    '3C01',
    '3F03',
    '3G03',
    '3G04',
    '3G08',
    '3G09',
    '6B10',
    '6C02',
    '6C03',
    '6C11',
    '6F04',
    '6H08',
    '7H11',
    'LA04']


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


factors = [
    'lane number',
    'temperature',
    'photoperiod',
    'genotype',
    'sample time',
    'replicate'
    ]


def read_faqiangs_slides(slides, text_file):
    """Read Tables 0, 1 and 2 from Faqiang's slides and file.
    
    Args:
        slides: Faqiang's slides.
        text_file: For lane 44.
    
    Returns:
        A 3-tuple of DataFrames with indices and column names resembling
        Faqiang's original slides.
    """
    tables = pd.read_excel(slides, sheetname='Tables')
    table_0 = pd.DataFrame(tables.iloc[0:8, 1:6])
    table_0.columns = tables.iloc[11, 1:6]
    table_0.columns.name = None
    table_0.index = tables.iloc[0:8, 0]
    table_0.index.name = None
    table_1 = pd.DataFrame(tables.iloc[12:20, 1:13])
    table_1.columns = tables.iloc[11, 1:13]
    table_1.columns.name = None
    table_1.index = tables.iloc[12:20, 0]
    table_1.index.name = None
    table_2 = pd.read_csv(text_file, sep=' ')
    return table_0, table_1, table_2


def sample_id_parser_deluxe(sample_id, table_0, table_1, table_2):
    """The deluxe version of sample ID parser for all 44 lanes of Spring
    2017 RNA-seq.

    Args:
        sample_id: Sample ID to be parsed.
        table_0: First table from Faqiang's slides.
        table_1: Second table from Faqiang's slides.
        table_2: Third table from Faqiang's slides.

    Returns:
        A tuple of lane, temperture, photoperiod, genotype, sample time,
        and replicate.
    """
    if sample_id[0:2] == 'YH':
        return sample_id_parser(sample_id, table_0, table_1)
    elif len(sample_id) <= 2:
        # Photothermal, genotype, time point and replicate.
        pgtr = table_2.loc[int(sample_id)]
        lane = 44
        pp, temp = pt_to_photo_temp(pgtr[0])
        genotype = pgtr[1]
        sample_time = sample_time_converter[pgtr[2]]
        replicate = pgtr[3]
        return lane, temp, pp, genotype, sample_time, replicate
    else:
        # Lane 41-43.
        if sample_id in lane_41_sample_ids:
            lane = 41
        elif sample_id in lane_42_sample_ids:
            lane = 42
        elif sample_id in lane_43_sample_ids:
            lane = 43
        else:
            print('Unknown sample ID {} in extra lanes.'.format(sample_id))
            exit(1)
        # Remove the underscores due to erroneous convertion to
        # E-notation in scientific notation.
        if '_' in sample_id:
            integer = re.search(r'^([0-9])_', sample_id).group(1)
            index = re.search(r'E_?([0-9]{2})', sample_id).group(1)
            converted_sample_id_real = integer+'E'+index
        # Correct LA04 to 0A04.
        elif sample_id == 'LA04':
            converted_sample_id_real = '0A04'
        else:
            converted_sample_id_real = sample_id
        # A prefix with a fake lane number is added.
        converted_sample_id = 'YH01_'+converted_sample_id_real
        _, temperature, pp, genotype, sample_time, replicate = (
            sample_id_parser(converted_sample_id, table_0, table_1)
            )
        return lane, temperature, pp, genotype, sample_time, replicate


def sample_id_parser(sample_id, table_0, table_1):
    """Sample ID parser.
    
    This parser is according to Faqiang's slides:
    'Sample ID info and explanation.xlsx'
    
    Args:
        sample_id: Sample ID as a string in the format YHnn_nAn(n),
            where n is a digit and A is a capital letter.
        table_0: Table 0 as a DataFrame in Faqiang's slides.
        table_1: Table 1 as a DataFrame in Faqiang's slides.

    Returns:
        A 6-tuple consisting of the following:
            * Lane number (1-44)
            * Temperature (16, 25 or 32)
            * Photoperiod ('LD', 'SD' or 'Sh')
            * Genotype (1-5)
            * Sampling time (
                one of the following strings:
                I1  D0  I3  I4  I5  I6
                    D1
                    D2
                    D3
                    D4
                II1 II2 II3 II4 II5 II6
                    D6
                    D7
                )
            * Replicate ('A' or 'B')
        Example return value: (13, 25, 'SD', 2, 'II2', 'A')
    """
    lane_number = int(sample_id[2:4])
    first_digit = sample_id[5]
    the_letter = sample_id[6]
    the_number = int(sample_id[7:])
    if first_digit == '0':
        temperature = 25
        # Use Table 0 to get the long ID with photoperiod condition.
        long_id_w_pp = table_0.at[the_letter, the_number]
        pp_single = long_id_w_pp[0]
        if pp_single == 'S':
            # Short day (SD) photoperiod condition.
            pp = 'SD'
        elif pp_single == 'L':
            # Long day (LD) photoperiod condition.
            pp = 'LD'
        else:
            print('Unknown photoperiod in sample ID parsing.')
            exit(1)
        long_id = long_id_w_pp[1:]
    else:
        if first_digit in ['3', '6', '8']:
            temperature = 16
        elif first_digit in ['2', '4', '9']:
            temperature = 25
        elif first_digit in ['1', '5', '7']:
            temperature = 32
        else:
            print('Unknown first digit in sample ID parsing.')
            exit(1)
        if first_digit in ['1', '3', '4']:
            pp = 'LD'
        elif first_digit in ['2', '5', '6']:
            pp = 'SD'
        else:
            # Must be in ['7', '8', '9'].
            pp = 'Sh'
        long_id = table_1.at[the_letter, the_number]
    sample_time = long_id[:-2]
    genotype = int(long_id[-2])
    replicate = long_id[-1]
    return lane_number, temperature, pp, genotype, sample_time, replicate


def pt_to_photo_temp(pt):
    """Convert photothermal to photoperiod and temperature.
    
    Args:
        pt: photoperiod indicator (1-9).
    
    Returns:
        A (photoperiod, temperature) 2-tuple.
    """
    photoperiod = ''
    temperature = ''
    if pt in [1, 3, 4]:
        photoperiod = 'LD'
    elif pt in [2, 5, 6]:
        photoperiod = 'SD'
    else:
        photoperiod = 'Sh'
    if pt in [1, 5, 7]:
        temperature = 32
    elif pt in [2, 4, 9]:
        temperature = 25
    else:
        temperature = 16
    return photoperiod, temperature


def anova_nfpkm_row(nfpkm_row, selector, table_0, table_1, table_2):
    """ANOVA for a single row of NFPKM.
    
    Args:
        nfpkm_row: A Series of the NFPKM values with and sample ID as
            index.
        selector: A list of 0-5, indicating the parsed features for
                grouping.
            0: Lane number.
            1: Temperature.
            2: Photoperiod condition.
            3: Genotype.
            4: Sample time.
            5: Replicate.
        table_0, table_1, table_2: Tables from Faqiang's slides.

    Returns:
        The p-value, the DOF between, the DOF within, and the fraction
            of explained variance..
    """
    # First run: calculate means.
    sum_group = {}
    count_group = {}
    sum_total = 0
    for x in range(nfpkm_row.size):
        group = select_feature(sample_id_parser_deluxe(
            nfpkm_row.index[x], table_0, table_1, table_2
            ), selector)
        value = nfpkm_row[x]
        if group in sum_group:
            sum_group[group] += value
            count_group[group] += 1
        else:
            sum_group[group] = value
            count_group[group] = 1
        sum_total += value
    mean_group = {group: sum_group[group]/count_group[group]
                  for group in sum_group}
    mean_total = sum_total/nfpkm_row.size
    # Second run: calculate variances.
    within_var = 0
    between_var = 0
    for x in range(nfpkm_row.size):
        group = select_feature(sample_id_parser_deluxe(
            nfpkm_row.index[x], table_0, table_1, table_2
            ), selector)
        within_var += (nfpkm_row[x]-mean_group[group])**2
        between_var += (mean_group[group]-mean_total)**2
    dof_within = nfpkm_row.count()-len(mean_group)
    dof_between = len(mean_group)-1
    f_stat = between_var / dof_between / within_var * dof_within
    #print('F({}, {}) stat:'.format(dof_between, dof_within), f_stat)
    explained_variance = 1/(1+dof_within/dof_between/f_stat)
    #print('Explained variance:', explained_variance)
    p_value = 1 - scipy.stats.f.cdf(f_stat, dof_between, dof_within)
    return p_value, dof_between, dof_within, explained_variance


def select_feature(feature, selector):
    """Select features from a tuple.
    
    Args:
        feature: A tuple of all features.
        selector: A list of indices indicating the features to be selected.

    Returns:
        A single feature or a tuple of features.
    """
    if len(selector) == 1:
        return feature[selector[0]]
    else:
        return tuple(feature[i] for i in selector)


def anova(data, rows, selector, table_0, table_1, table_2):
    """One-way ANOVA.
    
    Args:
        data: A DataFrame of data.
        rows: List of rows to calculate p-values for.
        selector: list of indices of factors to do the one-way ANOVA over.
        table_0, table_1, table_2: Tables from Faqiang's slides.
    
    Returns:
        Two lists of p-values and mean counts corresponding to rows.
    """
    result = [
        anova_nfpkm_row(
            data.iloc[row, :], selector, table_0, table_1, table_2
            ) for row in rows
        ]
    p_values = [x[0] for x in result]
    mean_counts = [data.iloc[row, :].mean() for row in rows]
    return p_values, mean_counts


def get_partition(data, partition, tables):
    """Get partition of data.

    Args:
        data: A DataFrame of data.
        partition: A list indicating the partition.
        tables: Tables from Faqiang's slides for sample ID parsing.

    Returns:
        A dictionary with partition tuples as the key and DataFrames as
        the values.
    """
    partitioned_data = {}
    for i in range(data.shape[1]):
        label = tuple([
            sample_id_parser_deluxe(
                data.columns[i], tables[0], tables[1], tables[2]
                )[z] for z in partition
            ])
        # A simple if expression is faster than setdefault() because no
        # accession is necessary.
        if label not in partitioned_data:
            partitioned_data[label] = pd.DataFrame()
        partitioned_data[label][data.columns[i]] = data.iloc[:, i]
    return partitioned_data


def load_data(data_file, quant_method='star'):
    """Load data from output of STAR, Kallisto or featureCounts.

    Args:
        data_file: Target file produced by the counting software.
        quant_method: Counting software; can be one of the following.
            * 'star'
            * 'kallisto'
            * 'feature_counts'

    Returns:
        A pandas DataFrame.

        The index is the default 0-based integers. The names of the
        columns are either extract from the data file (in the cases
        of Kallisto and featureCounts) or specified as module-level
        variables (in the case of STAR).
    """
    if quant_method == 'star':
        df = pd.read_csv(
            data_file, sep='\t', names=[
                gene_id_str, unstrand_str, fwstrand_str, rvstrand_str
                ], skiprows=4
            ).sort_values(by=gene_id_str)
    elif quant_method == 'kallisto':
        df = pd.read_csv(
            data_file, sep='\t'
            ).sort_values(by='target_id').reset_index(drop=True)
    elif quant_method == 'feature_counts':
        df = pd.read_csv(
            data_file, sep='\t', skiprows=1
            ).sort_values(by='Geneid')
    else:
        print('Unknown quantification method.')
        exit(1)
    return df


def extract_column(df, col_name='Length', icol=0):
    """Extract a column from a DataFrame and trim the gene IDs.

    Args:
        df: A DataFrame with long gene IDs in the first column.
        col_name (optional): The name of the column to be extract;
            default is 'Length'.
        icol (optional): The 0-based index for the column; overrides
            col_name if set to a positive integer.

    Returns:
        A Series of extracted column with shortened gene IDs as index.
    """
    # We trim the following suffix of the gene IDs to make them shorter
    # in the returned Series.
    size_gene_suffix = len('.Wm82.a2.v1')
    gene_ids = pd.Series(
        [gene[:-size_gene_suffix] for gene in df.iloc[:, 0]]
        )
    if icol:
        sr_col = df.iloc[:, icol]
    else:
        sr_col = df[col_name]
    sr_col.index = gene_ids
    return sr_col


def anova_tpm(star_count_file, feature_count_file, faqiang_slides,
              faqiang_txt, num_genes_requested, var_factor, par_factor):
    """ANOVA for TPM.

    Args:
        star_count_file: STAR count file.
        feature_count_file: featureCounts file.
        faqiang_slides: Faqiang's slides 'Sample ID info and
            explanation.xlsx' on sample ID mapping.
        faqiang_txt: Text file extracted from Faqiang's slides
            '3rd RNAseq samples_FW_23March2017.pptx'.
        num_genes_requested: Number of genes for ANOVA.
            -1 means all of the genes.
        var_factor: A list of integers indicating the variance factors.
        par_factor: A list of integers indicating the partition factors.

    Returns:
        A 2-tuple. The first element is a dictionary with partition index
            as the key and a tuple of two lists as the value, where the
            two lists are p-values and corresponding mean TPM values.
            The second element is a list of the gene IDs in the same order.
    """
    df_gene_exp = load_exp.load_exp(star_count_file, feature_count_file)
    table_0, table_1, table_2 = read_faqiangs_slides(
        faqiang_slides, faqiang_txt
        )
    if num_genes_requested == -1:
        num_genes = df_gene_exp.index.size
    elif num_genes_requested >= 0:
        num_genes = min(df_gene_exp.index.size, num_genes_requested)
    else:
        print('Invalid number of genes.')
        exit(1)
    rows = [i for i in range(num_genes)]
    tables = [table_0, table_1, table_2]
    df_tpm_dict = get_partition(df_gene_exp, par_factor, tables)
    anova_dict = {
        key: anova(
            df_tpm_dict[key], rows, var_factor, table_0, table_1, table_2
            ) for key in df_tpm_dict
        }
    gene_ids = list(df_gene_exp.index[rows])
    return anova_dict, gene_ids


def anova_lib_size(star_count_file, faqiang_slides, faqiang_txt,
                   var_factor, par_factor):
    """ANOVA for library sizes.

    Args:
        star_count_file: STAR raw count file.
        faqiang_slides: Faqiang's slides 'Sample ID info and
            explanation.xlsx' on sample ID mapping.
        faqiang_txt: Text file extracted from Faqiang's slides
            '3rd RNAseq samples_FW_23March2017.pptx'.
        var_factor: A list of integers indicating the variance factors.
        par_factor: A list of integers indicating the partition factors.

    Returns:
        A dictionary with partition index as the key and a tuple of
            p-value, the DOFs and the fraction of explained variance as
            the value.
    """
    df_counts = load_exp.load_and_drop(star_count_file)
    table_0, table_1, table_2 = read_faqiangs_slides(
        faqiang_slides, faqiang_txt
        )
    tables = [table_0, table_1, table_2]
    df_count_dict = get_partition(df_counts, par_factor, tables)
    anova_dict = {
        key: anova_nfpkm_row(
            df_count_dict[key].sum(), var_factor, table_0, table_1, table_2
            ) for key in df_count_dict
        }
    return anova_dict


def anova_header(key, par_factor, var_factor):
    """Header for ANOVA in the report.

    Args:
        key: A tuple indicating the levels of the partition factors.
        par_factor: Parition factor indices.
        var_factor: Variance factor indices.

    Returns:
        A string used as the header for a particular ANOVA subset.
    """
    if par_factor:
        par_text = ', '.join([
            '{} {}'.format(factors[i], key[it])
            for it, i in enumerate(par_factor)
            ])
    else:
        par_text = 'none'
    header = ('Partition: ' + par_text + 
              '.\nGroup by: ' +
              ', '.join([
                  factors[i] for i in var_factor
                  ]) +
              '.\n')
    return header


def sample_id_parser_compact(sample_id):
    """The compact version of the deluxe sample ID parser for Spring 2017
    RNA-seq data.

    Removed lane number and replicate from output since it is not
    relevant.

    Args:
        sample_id: Sample ID to be parsed.

    Returns:
        A tuple of temperture, photoperiod, genotype, and sample time.
    """
    faqiang_slides = "Sample ID info and explanation.xlsx"
    faqiang_txt = "lane-44-sample-id.txt"
    table_0, table_1, table_2 = read_faqiangs_slides(
        faqiang_slides, faqiang_txt
        )
    output = sample_id_parser_deluxe(sample_id, table_0, table_1, table_2)
    return output[1:5]


if __name__ == "__main__":
    main(sys.argv[1:])
