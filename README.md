# A Viewpoint Adaptation Ensemble Contrastive Learning Approach for Vessel Type Recognition with Limited Data
## Setup
Code was developed and tested on Ubuntu 22.04 with Python 3.8 and TensorFlow 2.8.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd VAECL-main/VAECL
pip3 install -r requirements.txt
```
## Download the DVTR Dataset
Run the following commands to download data sets from Google drive.
```
gdown https://drive.google.com/uc?id=132b9OeYS_lWbTjYuKXvmqIhPobCAREnq
unzip DVTR
rm DVTR.zip
```
## Download the trained generator
Run the following commands to download the trained generator.
```
cd c-wdcgan-gp
gdown https://drive.google.com/uc?id=1u8IDDmBvVMHenREeUkklGdhZ3RskqB7I
unzip save_gen
rm save_gen.zip
```
## Download the trained VAEL model
Run the following commands to download the trained VAECL model.
```
cd ../models
gdown https://drive.google.com/uc?id=1lXP8EOSj4HVGa3PAyExQ2h4MQKLZ9H1T
unzip save
rm save.zip
```
## Running Model
You can run the following command to replicate the results:
```
python3 vaecl.py
```
## Training the VAECL Model
You can run the following command to train the VAECL model.
```
python3 vaecl.py --train True
```
## Training the C-WDCGAN-GP Model
You can run the following command to train the C-WDCGAN-GP model.
```
cd ../c-wdcgan-gp
python3 train.py
```
