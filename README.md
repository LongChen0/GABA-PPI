# Proteomic Learning Studies of Gamma-Aminobutyric Acid (GABA) Receptor-Mediated Anesthesia
This script is for the paper "Proteomic Learning Studies of Gamma-Aminobutyric Acid (GABA) Receptor-Mediated Anesthesia". With the script, machine-learning regression model based on natural language processing (NLP) method can be built.
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
The autoencoder (AE) feature generation follows the work "Learning Continuous and Data-Driven Molecular Descriptors by Translating Equivalent Chemical Representations" by Robin Winter, Floriane Montanari, Frank Noe and Djork-Arne Clevert.
```
git clone https://github.com/jrwnter/cddd.git

cd cddd
conda env create -f environment.yml
source activate cddd
```
### BET fingerprint environment
The bidirectional encoder transformer (BET) feature generation follows the work "Extracting Predictive Representations from Hundreds of Millions of Molecules" by Dong Chen, Jiaxin Zheng, Guo-Wei Wei, and Feng Pan.
```
git clone https://github.com/WeilabMSU/PretrainModels.git

cd PretrainModels/bt_pro
mkdir bt_pro
mkdir bt_pro/fairseq
mkdir bt_pro/fairseq/data
python setup.py build_ext --inplace
mv ./bt_pro/fairseq/data/* ./fairseq/data/
```
## Generate molecular fingerprints
After creating the environment, the following code is used to generate AE fingerprints and BET fingerprints respectively.
```
Python generate_AE_feature.py
Python generate_BET_feature.py
```

## Build machine learning model with the molecular fingerprints
We built a machine learning model to predict binding affinity.
```
python BA_value_fei_SVM_con.py
```
## Reference
1. Chen, Dong, et al. "Extracting predictive representations from hundreds of millions of molecules." The journal of physical chemistry letters 12.44 (2021): 10793-10801.
2. Winter, Robin, et al. "Learning continuous and data-driven molecular descriptors by translating equivalent chemical representations." Chemical science 10.6 (2019): 1692-1701.

## License
All codes released in this study is under the MIT License.
