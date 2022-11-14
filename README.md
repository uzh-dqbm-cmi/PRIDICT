# PRIDICT: PRIme editing guide RNA preDICTion 

![PRIDICT logo](pridict_logo.png)

For accessing Supplementary Files, click [here](https://github.com/uzh-dqbm-cmi/PRIDICT/tree/supplementary_files).

Repository containing `python` package for running trained `PRIDICT` (PRIme editing guide RNA preDICTion) models. `prieml` package includes modules to setup and run `PRIDICT` models for predicting `prime editing efficiency and product purity` - [see demo below](#pridict-model-running-demo-%EF%B8%8F).

For more info about this research, see our [webapp](https://pridict.it/).

### Installation

* `git clone` the repo and `cd` into it.
* Run `pip install -e .` to install the repo's python package.

### Using Anaconda (optional) 🐍

The easiest way to install and manage Python packages on various OS platforms is through [Anaconda](https://docs.anaconda.com/anaconda/install/). Once installed, any package (even if not available on Anaconda channel) could be installed using pip. 

#### On Mac 

* Install [Anaconda](https://docs.anaconda.com/anaconda/install/).
* `git clone` the repo and `cd` into it.
* Start a terminal and run

    ```shell
    # create an environment
    conda create --name pridict
    # activate the created environment (pridict)
    conda activate pridict
    # install anaconda
    conda install -c anaconda python=3.8.5
    # update all installed packages
    conda update -y --all
    # install pytorch 
    # Note cudatoolkit version depends on the version installed on your device
    # if there is no GPU run this command
    # conda install pytorch torchvision -c pytorch
    # see https://pytorch.org/
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    conda clean -ya
    # install prieml package (i.e. this package)
    pip install -e .
    ```
* Now we are ready to use the package with the trained `PRIDICT` models for prime editing prediction.

### Running PRIDICT in shell 🔲

- Running in `manual` model, where we supply
  -  `--sequence-name`: name of the sequene (i.e. unique id for the sequence)
  -  `--sequence`: target sequence to edit in quotes (format: `"xxxxxxxxx(a/g)xxxxxxxxxx"`; minimum of 100 bases up and downstream of brackets are needed)
  -  `--output-dir`: output directory where results are dumped on disk
  -  `--use-5folds`: Use all 5-folds trained models. Default is to use fold-1 model
```shell

python pridict_pegRNA_design.py manual --sequence-name seq1 --sequence 'GCCTGGAGGTGTCTGGGTCCCTCCCCCACCCGACTACTTCACTCTCTGTCCTCTCTGCCCAGGAGCCCAGGATGTGCGAGTTCAAGTGGCTACGGCCGA(G/C)GTGCGAGGCCAGCTCGGGGGCACCGTGGAGCTGCCGTGCCACCTGCTGCCACCTGTTCCTGGACTGTACATCTCCCTGGTGACCTGGCAGCGCCCAGATGCACCTGCGAACCACCAGAATGTGGCCGC' --output-dir ./predictions_manual
``` 

- Running in `batch` model, where we supply
  -  `--input-dir` : directory where the input csv file is found on disk
  -  `--input-fname`: input file name - name of csv file that has two columns [`editseq`, `sequence_name`]. See `batch_template.csv` in the `./input` folder
  -  `--output-dir`: directory on disk where to dump results
  -  `--output-fname`: output filename used for the saved results
  -  `--combine-results`: Compile all results in one dataframe
  -  `--use-5folds`: Use all 5-folds trained models. Default is to use fold-1 model
```shell

 python pridict_pegRNA_design.py batch --input-dir ./input/ --input-fname batch_example_file.csv --output-dir ./predictions_batch --output-fname batchseqs

``` 

### PRIDICT model running demo 🏃‍♀️

We provide two `notebooks` that illustrate the use of two `PRIDICT` models on `processed datasets`: one trained on [our library]() and another using [`Kim et al. library`](https://doi.org/10.1038/s41587-020-0677-y) under `notebooks` folder:
   - [`PRIDICT` trained on `our library` & tested on `Kim et al. library`](/notebooks/Prieml_outcomedistrib_trained_schwank_predict_hyongbum.ipynb)

### OS & Packages' version

The models were trained, tested and ran on Linux machine `Ubuntu 18.04.3 LTS` with one `Tesla P4 GPU` support.
The version of the `required` packages used in `setup.py` were:
* `numpy` &gt;=  `'1.19.2'`
* `scipy` &gt;= `'1.5.3'`
* `pandas` &gt;= `'1.0.1'`
* `scikit-learn` &gt;= `'0.23.2'`
* `torch` &gt;= `'1.7.1'`
  * `cudatoolkit=10.1`
* `matplotlib` &gt;= `'3.3.4'`
* `seaborn` &gt;= `'0.11.1'`
* `prettytable` &gt;= `'2.1.0'`
* `tqdm` &gt;= `'4.64.0'`

### Webapp 🕸️

A running instance of the package for `optimizing prime editing guide RNA design` and `predicting prime editing efficiency and product purity` can be accessed at this [link](https://pridict.it/).

### Citation

If you find our work is useful in your research, please cite the following paper:

> @article {Mathis et al.,  
	author = {Mathis, Nicolas and Allam, Ahmed and Kissling, Lucas and  Marquart, Kim Fabiano and Schmidheini, Lukas and Solari, Cristina and Balázs, Zsolt and Krauthammer, Michael and Schwank, Gerald},  
	title = {Predicting prime editing efficiency and product purity by deep learning},  
	year = {2022},  
	doi = {},  
	URL = {},  
	eprint = {},  
	journal = {}  
}
