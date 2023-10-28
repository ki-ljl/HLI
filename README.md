# Hierarchical Label Inference Incorporating Attribute Semantics in Attributed Networks (ICDM 2023)
Code for paper---**H**ierarchical **L**abel **I**nference Incorporating Attribute Semantics in Attributed Networks.
![在这里插入图片描述](https://img-blog.csdnimg.cn/56c901568d5e4fa9bee5abf62181a315.png#pic_center)
## Code Overview
```bash
HLI:.
│  args.py
│  get_data.py
│  pytorchtools.py
│  requirements.txt
│          
├─data            
│  ├─meta_Arts_Crafts_and_Sewing
│  │      
│  └─meta_Industrial_and_Scientific
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
6. **src/main.py**: Public opinion concern prediction.
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
Due to the file size limitation of github, we uploaded the datasets to [Google Drive](https://drive.google.com/drive/folders/1Kbt2B8qEyG48A-TIbxO5bvpBuziC8DDd?usp=sharing). So download the dataset from Google Drive before you run the code.
### Full-supervised Attribute Label Inference
```bash
cd src/
python main.py --experiment full --dataset meta_Industrial_and_Scientific
python main.py --experiment full --dataset meta_Arts_Crafts_and_Sewing
```
### Semi-supervised Attribute Label Inference
```bash
cd src/
python main.py --experiment semi --dataset meta_Industrial_and_Scientific
python main.py --experiment semi --dataset meta_Arts_Crafts_and_Sewing
```
### Ablation Experiment
```bash
cd src/
python main.py --experiment full --dataset meta_Industrial_and_Scientific --use_hiera_att False
python main.py --experiment full --dataset meta_Industrial_and_Scientific --use_sibling_att False
python main.py --experiment full --dataset meta_Industrial_and_Scientific --use_sfc False
python main.py --experiment full --dataset meta_Industrial_and_Scientific --use_slp False
```
# Cite
