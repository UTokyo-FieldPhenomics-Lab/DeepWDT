![alt text](docs/DeepWDT.png)

This repository is the official implementation of DeepWDT (Deep Waggle Dance Translator), the most efficient and easy-to-use deep learning framework to detect and decode waggle dances.

## <div align="center">Requirements</div>

We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n deepwdt python=3.8
```

Then, activate the environment:
```Shell
conda activate deepwdt
```

Then, set up your working environment ot our repository:
```Shell
cd path_to_DeepWDT
```

Then install requirements:
```Shell
pip install -r requirements.txt 
```

## <div align="center">Dataset preparation</div>

### Prepare our dataset

To prepare our training set, copy the 'videos' folder and 'annotations.txt' into 'data/training_dataset' and launch the following command line:
```Shell
python src/dataset/unpack.py 
```

The training dataset can be downloaded at *link coming soon*.  

Feel free to augment the dataset with your own data to improve the model's generalizability.

### Data collection advices

*Coming soon.*

## <div align="center">Training and Validation</div>

We built our system from the original YOWOv2 network that can be found here : https://github.com/yjh0410/YOWOv2.

### Training
To retrain the network on the training dataset, you can use the following command line:

```Shell
python src/train.py path_to_the_config_file
```

The config file can be found at src/config/parameters.yaml in our repository.

We recommend to use MLflow to follow the training metrics. To do this, set "MLFLOW" to "true" in the config file and launch the MLflow session with:

```Shell
mlflow ui
```

### Validation

Validation metrics can be computed during training if set to "true" in the config file.

To run the validation separately use the following command line:

```Shell
Coming Soon
```

## <div align="center">Inference</div>

### Infer on a new dataset

*Coming soon.*

### Translation to geographic coordinates

*Coming soon.*

### Use outputs from the model to augment the training dataset

*Coming soon.*

## <div align="center">License</div>

Our software is released under the MIT license.

## <div align="center">Contact</div>

DeepWDT is developed and maintained by Sylvain Grison (sylvain.grison@fieldphenomics.com).
