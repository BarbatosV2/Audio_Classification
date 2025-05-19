# Audio Classification (Machine Learning)

# Install Requirements
```
pip install -r requirements.txt
```

if the device python is 3.8 or older, need to install websocket
```
pip install websockets
```

# Create a directory for the dataset
```
mkdir -p audio_data
cd audio_data
```

# Download the dataset
Depends on what dataset u want to train. For me, I am training cat and dog dataset. Downloaded from following link.
```
https://zenodo.org/records/3563990
```

# Extract the Audio

Extract the audio folders and put both cat and dog wav files into the audio_data

# CSV File

CSV labeling Example

| filename | label |
| --- | --- |
| cat_0.wav | cat |
| dog_0.wav | dog |

or 

| filename | label |
| --- | --- |
| cat_0.wav | cat |
| dog_0.wav | dog |
| Null_0.wav | no_sound |

In VS Code
```
filename, label
cat_0.wav, cat
dog_0.wav, dog
```

or 

```
filename, label
cat_0.wav, cat
dog_0.wav, dog
Null_0.wav, no_sound
```

Cannot be label as null

# Traing

In utils.py, LABELS = ['name', 'another', 'obj'] need to be edited before start training

```
python train.py --batch 32 --workers 16 --device 0 --epoches 500 audio_data labeled_data.csv
```

batch, workers and epoches are changable according to machine can handle. 

## Source
For build in Microphone or webcam
```
python predict.py --source 0 --device 0 sound_model.pth
```
For local wav file
```
python predict.py --source ./audio_data/cat_0.wav --device 0 sound_model.pth
```
For link 
```
python predict.py --source https://audio_link/audio.wav --device 0 sound_model.pth
```
For webserver
```
python predict.py --source stream:ws://172.20.10.2:82 --device 0 sound_model.pth
```

# Predicting 

## WAV File Prediction
```
python predict.py --source ./audio_data/dog_5.wav --device 0 sound_model.pth
```

## For Ordinary Prediction
```
python predict.py --source 0 --device 0 sound_model.pth
```
## For Null label Prediction
```
python predict.py --source 0 --device 0 --null_label "silence" sound_model.pth
```
## For Confidencial threshold Prediction
```
python predict.py --source 0 --device 0 --confidence_threshold 0.8 --null_label "silence" sound_model.pth
```

And then it will save detect_1.txt and contonuous_rec_1.wav in recorded folder.
When predicting the model, if the accuracy is under 80%, it will show as null, otherwise cats or dogs (as trained in cat and dog labels)
This accuracy value can be changed under predict.py or by simplying changing --confidence_threshold number.
Training with no_sound or natural sound will be fine as well. 

P.S Make sure micrphone is ok for the better accuracy. 


## hz_classifier.py

This code is to check the Hz rate frequency of wav file
```
python hz_classifier.py ./audio_data/cat_0.wav
```

Happy Coding
