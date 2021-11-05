#!/usr/bin/env bash
#
# Network reconstruction using simulated biologically plausible data.

######################################################################
# Simulate the planted edge model and run CausNet_BSLR.
# Change parameters in bio-data-gen.py.  Assume the required packages
# are loaded in the virtual environment.
#
# Args:
#     None
#
# Returns:
#     A GraphML format file named grn.xml.
######################################################################

# Generate the biologically plausible data using the planted-edge
# model.
nice time ./bio-data-gen.py
# Run CausNet.
nice time ./soybean.py \
     -c cond-list-file.txt \
     -p 100 `# Number of perturbations.` \
     -i gene-list.csv \
     -g grn.xml `# Output file.` \
     -x exp.csv `# Expression file.` \
     -P design.csv `# Design file.` \
     -l 1 `# Number of time lags.`
