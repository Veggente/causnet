# soybean-network
A network analysis protocol for soybean

# Python environment setup
## For Windows users
1. Download [`Miniconda` installer](https://conda.io/miniconda.html) with `Python 3` for Windows.

2. Open the Anaconda prompt and create a virtual environment with `matplotlib`, `networkx`, `scipy` and `tqdm` packages.
```
> conda create --name myvenv matplotlib networkx scipy tqdm
```

3. Activate the environment
```
> activate myvenv
(myvenv) >
```
Notice `(myvenv)` at the beginning of the prompt indicates the environment named `myvenv` is active.
## For Unix users (macOS and Linux)
1. Install Python 3.

2. Setup virtual environment.
```
$ python3 -m venv myvenv
$ source myvenv/bin/activate
(myvenv) $
```

3. Install packages.
```
(myvenv) $ cd scripts/
(myvenv) $ pip install -r requirements.txt
```

# Network analysis
1. Obatin the normalized gene expression file `expression-2011.csv`.
1. Generate the gene list file.

    Create a spreadsheet with two columns using Excel or Numbers. The first column is the list of the gene IDs and the second column is the list of the gene names. Then save the table as `gene-list.csv` in CSV (comma-separated values) format. An example of the generated CSV file looks like this.
    
    `gene-list.csv`:
    
    ```
    Glyma.19G224200,GmPHYA (E3)
    Glyma.20G090000,GmPHYA (E4)
    Glyma.06G196200,GmTOC1a
    Glyma.19G260900,GmLHY
    ```
1. Run the gene regulatory network inference algorithm.
    ```
    (myvenv) > python soybean.py -x expression-2011.csv -i gene-list.csv -g grn.xml -p 10 -c cond_list_file.txt
    ```
    Let's break it down:
    * Option `-x`: Use the expression file `expression-2011.csv`.
    * Option `-i`: Use the gene list file `gene-list.csv`.
    * Option `-g`: Save the inferred network in file `grn.xml`.
    * Option `-p`: Aggregate 10 perturbed runs in the inferred network. Larger number of perturbations gives more reliable network reconstruction, but also takes more time to compute. On a laptop 10 perturbations of a 40-gene network take ~5 minutes.
    * Option `-c`: Condition list file. This is a JSON format file of a list of lists
                    specifying the conditions of the samples to do
                    network analysis on. The order of the lists should
                    be compatible with the parser.
    
    Note if the output file `grn.xml` already exists, it will be overwritten.
    
    Some other options are as follows.
    * Option `-r`: Seed for the random number generator.
    * Option `-l`: The number of time lags for network inference. The default is 1.
    * Option `-m`: The maximum in-degree ofthe network. The default is 3.
    * Option `-f`: The significance level for edge rejection based on Granger causality. The default is 0.05.
    * Option `-u`: Input file to specify gene clusters.
    * Option `-v`: Virtual time shift: replicate the first time and append it to the end in order to close the loop from the last time to the first time the next day.

1. Open `grn.xml` with `Cytoscape`.
    1. Install and open `Cytoscape`.
    1. Open network from network file `grn.xml`.
    1. Go to `File -> Import -> Styles... ` and select `styles-soybean.xml` to import the style.
    1. Go to `Control panel -> Style` and select `soybean` to apply the style.
    
    Note steps iii and iv load the network style and only need to be done once for a session.

    You can change the layout of the network by, e.g., going to `Layout -> Attribute Circle Layout -> shared name`.
    
    Other functionalities of `Cytoscape` can be found in [its manual](http://manual.cytoscape.org/en/stable/).
