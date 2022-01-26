# CBIR-Mars
Content-based image retrieval of mars images as part of Computer Vision for Remote Sensing at TU Berlin

## Setup

To install, clone the git repository at https://git.tu-berlin.de/rsim/cv4rs-2021-summer/CBIRforMars.  The code was written for Python 3.9.5.  Required libraries that need to be installed with pip can be seen in the following table:

| library | version |
| ------- | --- |
| torch | 1.8.1+cu111 |
| torchvision | 0.9.1+cu111 |
| tqdm | 4.60.0 |
| pytorch_metric_learning | 0.9.99 |
| numpy | 1.20.2 |
| Pillow | 8.2.0 |
| cupy-cuda111 | 9.0.0 |
| matplotlib | 3.4.1 |
| sklearn | 0.24.2 |

However, we expect that for most of these libraries the most current version should not pose problems.  The only libraries where care might need to be taken are torch, torchvision and cupy-cuda.  These need to be installed according to the cuda version your graphics card supports and they also have to match up to each other.  Also for pytorch_metric_learning, we recommend using the exact version given. Lastly you need to copy the data set into the root folder.  The data set is provided at https://tubcloud.tu-berlin.de/s/fYfckqqW2W2GwgP.  You can also download the original DoMars16k data set at https://zenodo.org/record/4291940, we simply split up the train folder into a train and a database folder using a 50/50 split.

## Usage
The following commands need to be executed in /geomars-pytorch for the MIRS model and in /basic-pipeline for the baseline models and the training of the feature extractors. python refers to your local python3 command.

To train a network first setup the desired configuration of hyperparameters in hparams.py and then run:
```python
python train.py
```
After training is completed the new model is now stored in /outputs/model_best.pth. To build an new feature/hashcode archive from this model run:
```python
python build_db.py
```
When the generation of the archive is completed queries can now be made with:
```python
python query.py data/test/PATH_TO_IMAGE
```
For more information about the usage of the system and more technical information, refer to "Technical Documentation.pdf".
