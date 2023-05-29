# STAN: Segmentation Transformer Age Network

## Setup

Please install the required packages in “requirements.txt” before running the scripts. We recommend you to use the command shown at [PyTorch.org](https://pytorch.org/get-started/locally/) to install the suitable version of PyTorch. 

## Parameter Settings

You can set the parameters for testing in “config.py”, which include the paths of datasets, the hyperparameters of the model, and the type of objective functions. Please refer to the annotations in “config.py” for more details. The parameters will be saved automatically in training at `output_dir + hyperparameter.json` .

## Training

### Stage One

The recommended learning rate and the number of training epochs are 1e-4 and 200. 
After setting the data paths of datasets at “config.py”, you can run `python train_seg.py `. 

### Stage Two

After the training of Stage One, please input the path of the best model in “–pretrained_model” at “config.py”. 
The recommended learning rate and the number of training epochs are 1e-5 and 400. 
You can run `python train.py`  for training at Stage Two.

## Testing

### Age Estimation

Please run `python predict.py`  after setting the name of the best model in “–model_name” at “config.py”. 
The results of age estimation will be saved at `output_dir + prediction.txt` . The results of segmentation will be saved at `infer_dir`.

### Bias Correction

Please run `python bias_correction.py`  after setting the name of the best model in “–model_name” at “config.py”. 
The results of bias correction will be saved at `output_dir + bias_correction.txt` .

### Segmentation Evaluation

Please run `python evaluation.py` after setting the paths of ground-truth and outputs in the script.
The evaluation results will be saved at `InferPath + /eval_xxx.txt` .