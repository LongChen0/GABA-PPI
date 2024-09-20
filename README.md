# Proteomic Learning Studies of Gamma-Aminobutyric Acid(GABA) Receptor-Mediated Anesthesia
This script is for the paper "Proteomic Learning Studies of Gamma-Aminobutyric Acid(GABA) Receptor-Mediated Anesthesia". With the script, machine-learning regression model based on natural language processing (NLP) method can be built.
## Requirements
OS Requirements
* CentOS Linux 7 (Core)

Python Dependencies
* setuptools (>=18.0)
* python (>=3.7)
* pytorch (>=1.2)
* rdkit (2020.03)
* biopandas (0.2.7)
* numpy (1.17.4)
* scikit-learn (0.23.2)
* scipy (1.5.2)
* pandas (0.25.3)
* cython (0.29.17)
## Download the repository
Download the repository from Github
```
# download repository by git  
git clone https://github.com/LongChen0/GABA-PPI.git
```
## Install the pretrained model for molecular fingerprint generation
### AE fingerprint environment
The AE feature generation follows the work "Learning Continuous and Data-Driven Molecular Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.
```
git clone https://github.com/jrwnter/cddd.git

cd cddd
conda env create -f environment.yml
source activate cddd
```
### BET fingerprint environment
The BET feature generation follows the work "Extracting Predictive Representations from Hundreds of Millions of Molecules" by Dong Chen, Jiaxin Zheng, Guo-Wei Wei, and Feng Pan.
```
git clone https://github.com/WeilabMSU/PretrainModels.git

cd PretrainModels/bt_pro
mkdir bt_pro
mkdir bt_pro/fairseq
mkdir bt_pro/fairseq/data
python setup.py build_ext --inplace
mv ./bt_pro/fairseq/data/* ./fairseq/data/
```
