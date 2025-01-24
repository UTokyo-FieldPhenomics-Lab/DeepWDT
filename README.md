![alt text](docs/DeepWDT.png)

This repository is the official implementation of DeepWDT (Deep Waggle Dance Translator), the most efficient and easy-to-use deep learning framework to detect and decode waggle dances.

## <div align="center">Requirements</div>

We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n deepwdt python=3.9
```

Then, activate the environment:
```Shell
conda activate deepwdt
```

Then install requirements:
```Shell
pip install -r requirements.txt 
```

## <div align="center">Dataset preparation</div>

To prepare our training set, copy the 'videos' folder and 'annotations.txt' into 'data/training_dataset' and launch the following command line:
```Shell
python src/dataset/prepare_training_dataset.py -videos 
```
The most recent weights are from training on a version of the dataset that was resized to half the original size (224x224 instead of 448x448 as mentioned in our paper).

The training dataset can be dowloaded at:

You can freely add your own annotated videos to the training dataset to enhance the generalizability of the model.

## <div align="center">Dance Recognition Training and Validation</div>

For the detection of the dancing individuals, we adapted the original YOWOv2 network that can be found here : https://github.com/yjh0410/YOWOv2.

### Training
To retrain the network on the training dataset, you can use the following command line (example for K=8):

```Shell
python -m src.training.train_recognition --version yowo_v2_nano --max_epoch 20 --len_clip 8
```

Weights are saved in 'runs/training/weights'.

You can also train models in a batch with multiple K values at the same time using: 'batch_train_recognition.sh'.

### Validation

To obtain the frame mAP from the validation set you can use the following command line (example for K=8 and epoch=20):
```Shell
python -m src.evaluator.eval_recognition --version yowo_v2_nano --len_clip 8 --img_size 224 --cal_frame_mAP --eval_split val --epoch 20
```

To obtain the video mAP from the validation set you can use the following command line (example for K=8 and epoch=20):
```Shell
python -m src.evaluator.eval_recognition --version yowo_v2_nano --len_clip 8 --img_size 224 --cal_frame_mAP --eval_split val --epoch 20
```

Results are saved in 'runs/evaluation/recognition'.

You can also run multiple combinations at the same time using: 'batch_eval_recognition.sh'.

## <div align="center">Dance Tracking Validation</div>

To obtain the metrics from the validation set you can use the following command line (example for K=8 and epoch=20):
```Shell
python -m src.evaluator.eval_tracking --version yowo_v2_nano --len_clip 8 --eval_split val --epoch 20
```

You can also run multiple combinations at the same time with: 'batch_eval_tracking.sh'.

## <div align="center">Inference</div>

### Infer on a new dataset

To prepare a new datasets for inference, you have to use the following command line first:
```Shell
python src/dataset/add_dataset.py --dataset name_of_the_new_dataset
```
and then copy your videos to 'dataset/name_of_the_new_dataset/videos'.

Then, to infer on this new dataset you can use the following command line (example):
```Shell
python -m src.inference.infer --version yowo_v2_nano --dataset name_of_my_dataset --video_format mp4 --len_clip 8 --img_size 960,540 --min_duration 55 --ext_tool labelme --result_video
```

The weights will be sourced from 'runs/training/weights'. The --img_size parameter will automatically resize videos to the nearest multiple of 32. Our latest weights available were trained on 224x224 video frames instead of the 448x448 ones used in the paper (448x448 video frames clipped from 1920x1080 videos and downsampled to 224x224). If you're using videos of a similar resolution (1920x1080 with a similar zoom level), it's recommended to process them similarly and set --img_size to half the size of your video resolution when infering.

### Translation to geographic coordinates

(!) This part is under development.

### Use outputs from the model to augment the training dataset

Outputs are automatically converted to the labelme format at 'runs/inference/name_of_my_dataset/labelme' if you use the parameter --labelme during inference. You can freely correct these annotations and incorporate them back to the training dataset.

## <div align="center">License</div>

Our software is released under the MIT license.

## <div align="center">Contact</div>

DeepWDT is developed and maintained by Sylvain Grison (sylvain.grison@fieldphenomics.com).
