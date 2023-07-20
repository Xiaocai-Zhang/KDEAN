# Knowledge-Based Dual External Attention Network for Peptide Detectability Prediction
## Setup
Code was developed and tested on Ubuntu 22.04 with Python 3.8 and TensorFlow 2.8.0. You can setup a virtual environment by running the code like this:
```
python3 -m venv env
source env/bin/activate
cd KDEAN-main/KDEAN
pip3 install -r requirements.txt
```
## Download the trained models
Run the following commands to download the trained VAECL model.
```
gdown https://drive.google.com/file/d/1pDUONi_ELMnZGBqI8uNGQq44ZJRn10DH/view?usp=drive_link
unzip Model
rm Model.zip
```
If teh download is not working, please use the following link to manually download the models 
```
https://drive.google.com/file/d/1pDUONi_ELMnZGBqI8uNGQq44ZJRn10DH/view?usp=drive_link
```
## Perfromance Test with the Trained Models
You can run the following command to replicate the results:
```
python3 main.py --dataset "Mus Musculus"
```
Similarily, you can change the dataset to see the perfoamnce on other datasets.
## Train the KDEAN Model
You can run the following command to train the VAECL model.
```
python3 main.py --dataset "Mus Musculus"--train True
```
If you want to use GPU accelerated computing, you can call
```
python3 main.py --dataset "Mus Musculus"--train True --GPU True
```
