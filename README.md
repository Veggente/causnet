![CausNet](/logo/causnet-144ppi.png)
# CausNet
A Causal Inference Algorithm for Gene Regulatory Network Reconstruction.

## Introduction
CausNet recovers the gene regulatory network from time-series gene expression data.

## Python environment setup
### Quick install (recommended for Windows/macOS/Linux)
1. Download [`Miniconda` installer](https://conda.io/miniconda.html) with `Python 3.x` for Windows, macOS or Linux and install.

1. Create a virtual environment with `matplotlib`, `networkx`, `scipy`, `tqdm`, `pandas` and `xlrd` packages by typing the following in the Anaconda prompt for Windows, or in the terminal for macOS and Linux:
    ```sh
    > conda create --name myvenv matplotlib networkx scipy tqdm pandas xlrd
    ```

1. Activate the environment.
    * For Windows, type
        ```sh
        > activate myvenv
        ```
    * For maxOS or Linux, type
        ```sh
        $ source activate myvenv
        ```
    Notice there will be a `(myvenv)` at the beginning of the prompt indicating the environment named `myvenv` is active.

### Full install (optional for macOS/Linux)
1. Install Python 3.

1. Set up the virtual environment.
    ```sh
    $ python3 -m venv myvenv
    $ source myvenv/bin/activate
    (myvenv) $
    ```

1. Install required packages.
    ```sh
    (myvenv) $ pip install -r requirements.txt
    ```

## Network analysis
1. Generate the file containing the normalized expression levels of all the genes for your experiment in CSV format.

    The first row is the list of sample IDs and the first column is the list of Gene IDs.  Other entries are the normalized gene expression levels.  See an example in `expressions.csv`.

1. Generate your experiment design file.

    The first column is the list of sample IDs, and the other columns are the conditions for the experimental factors.  Note the last factor must be the sample time.  At least two replicates per experimental condition is required.  See an example in `design.csv`.

1. Generate your gene list file for the reconstruction.

    Create a spreadsheet with two columns using Excel, Numbers or other spreadsheet applications.  The first column is the list of the IDs for the genes you want to reconstruct the network for, and the second column is the list of the gene names to show up in the network file.  Then save the table as a file in CSV (comma-separated values) format.  See an example in `gene-list.csv`.

1. Generate your condition list file for the reconstruction.

    Create a text file in JSON format of several lists, where each list specifies the levels to include in an experimental condition.  The last list must be ordered sample times.  See an example in `cond-list-file.txt`.

1. Run the gene regulatory network inference algorithm.  Note there is no space after the line-continuing backslash.
    ```sh
    (myvenv) > python causnet.py \
        -x expressions.csv \
        -P design.csv \
        -i gene-list.csv \
        -c cond-list-file.txt \
        -p 10 \
        -g grn.xml
    ```
    Let's break it down:
    * Option `-x`: Use the expression file `expressions.csv`.
    * Option `-P`: Use the design file `design.csv`.
    * Option `-i`: Use the gene list file `gene-list.csv`.
    * Option `-c`: Use the condition list file `cond-list-file.txt`.
    * Option `-p`: Aggregate 10 randomly perturbed runs in the inferred network.  Larger number of perturbations gives more accurate estimation of the reliability of the network reconstruction, but also takes more time to compute. On a laptop 10 perturbations of a 40-gene network take ~5 minutes.
    * Option `-g`: Save the inferred network as file `grn.xml` in GraphML format.
    
    Note the output file `grn.xml` will be overwritten if it already exists.
    
    Some other options are as follows.
    * Option `-r`: Seed for the random number generator.
    * Option `-l`: The number of time lags for network inference. The default is 1.
    * Option `-m`: The maximum in-degree ofthe network. The default is 3.
    * Option `-f`: The significance level for edge rejection based on Granger causality. The default is 0.05.
    * Option `-v`: Virtual time shift: replicate the first times and append them to the end in order to close the loop from the last time to the first time the next day.  The default is 0.

1. Open `grn.xml` with `Cytoscape`.
    1. Install and open `Cytoscape`.
    1. Open network from network file `grn.xml`.
    1. Go to `File -> Import -> Styles... ` and select `style-causnet.xml` to import the style.
    1. Go to `Control panel -> Style` and select `causnet` to apply the style.
    
    Note steps iii and iv load the network style and only need to be done once for a `Cytoscape` session.

    You can change the layout of the network by, e.g., going to `Layout -> Attribute Circle Layout -> shared name`.
    
    Other functionalities of `Cytoscape` can be found in [its manual](http://manual.cytoscape.org/en/stable/).
