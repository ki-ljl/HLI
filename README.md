# Hierarchical Label Inference Incorporating Attribute Semantics in Attributed Networks (ICDM 2023)
Code for paper---**H**ierarchical **L**abel **I**nference Incorporating Attribute Semantics in Attributed Networks.
![image](https://github.com/ki-ljl/HLI/assets/56509367/8d60f6a1-9759-4089-8e4b-1dbf525ec215)

## Code Overview
```bash
HLI:.
│  args.py
│  get_data.py
│  pytorchtools.py
│  requirements.txt
│          
├─data            
│  ├─ACS
│  │      
│  └─IS
│          
└─src
    ├─main.py
    ├─models.py
    └─util.py
```
1. **args.py**: args.py is the parameter configuration file, including model parameters and training parameters.
2. **get_data.py**: This file is used to load the data.
3. **pytorchtools**.py: This file is used to define the earlystopping mechanism.
4. **requirements.txt**: Dependencies file.
5. **data/**：Dataset folder.
6. **src/main.py**: Main file.
7. **src/models.py**: Implementation of HLI.
8. **src/util.py**: Defining various toolkits.
## Dependencies
Please install the following packages:
```
gensim==4.1.0
matplotlib==3.2.2
networkx==2.3
numpy==1.21.6
pandas==1.2.3
scikit-learn==1.0.2
scipy==1.7.3
torch==1.10.1+cu111
torch-cluster==1.5.9
torch-geometric==2.2.0
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
tqdm==4.62.3
```
You can also simply run:
```
pip install -r requirements.txt
```
## Usage
### Dataset Preparation
Due to the file size limitation of github, we uploaded the datasets to [Google Drive](https://drive.google.com/drive/folders/1Kbt2B8qEyG48A-TIbxO5bvpBuziC8DDd?usp=sharing). Please download the two **hash_vector_embeddings_2048.pkl** files and place them in the corresponding folders before running the code.
### Full-supervised Attribute Label Inference
```bash
cd src/
python main.py --experiment full --dataset IS
python main.py --experiment full --dataset ACS
```
![image](https://github.com/ki-ljl/HLI/assets/56509367/e3797faf-2e46-40c4-89a7-2e6c45eeb56a#pic_center)

### Semi-supervised Attribute Label Inference
```bash
cd src/
python main.py --experiment semi --dataset IS
python main.py --experiment semi --dataset ACS
```
![image](https://github.com/ki-ljl/HLI/assets/56509367/8769e173-645a-44f1-a632-a6eb76f3d0ec)

# Cite
```
@INPROCEEDINGS{10415698,
  author={Li, Junliang and Yang, Yajun and Hu, Qinghua and Wang, Xin and Gao, Hong},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)}, 
  title={Hierarchical Label Inference Incorporating Attribute Semantics in Attributed Networks}, 
  year={2023},
  volume={},
  number={},
  pages={1091-1096},
  keywords={Semantics;Data models;Data mining;Periodic structures;attribute inference;hierarchical inference;label semantics},
  doi={10.1109/ICDM58522.2023.00129}
}
```
