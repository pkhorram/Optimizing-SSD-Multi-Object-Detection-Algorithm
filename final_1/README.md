# Optimizing SSD Multi-Object Detection Model (PyTorch)

## Table of contents
* [Installation](#Installation)
* [Datasets](#Datasets)
* [Demo](#Demo)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Performance](#Performance)
* [Experiments](#Experiments)
* [Directory structure](#Directory-structure)
* [References](#References)

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation

To install Python dependencies and modules, use <br>
```pip install -r requirements.txt``` <br>

- Choose your environment on the website and install [PyTorch](http://pytorch.org/) by running the appropriate command.
- Clone this repository.
- Install one of Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.

## Datasets

In this project we are using the `2007 and 2012 VOC trainval` dataset in order to train the model. Testing the model is implemented using `2007 VOC test` dataset. You can download the dataset by clicking on the following links. 

[2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)


[2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 

### Demo
(((NEEDS CHANGES)))
Run **SSD_Demo.ipynb** notebook to run Single-Shot Detection on a random image from the VOC2007 dataset. 

Update ` voc_root ` and ` dataset_root ` in config_vgg16_ssd.py with the location of your downloaded dataset.
`voc_root` -> should contain VOC2007 and VOC2012 directories -> ` PATH_TO/ ` 
and 
`dataset_root ` -> `PATH_TO/VOC2007/` 
### Training
Run **SSD_train.ipynb** notebook to train the SSD model on the PascalVOC2012+07 dataset.
### Evaluation
Run **SSD_Eval.ipynb** notebook to evaluate the SSD model on the VOC2007 test set.

Run **SSD_Eval_Testset.ipynb** notebook to evaluate the SSD model on the PascalVOC2007 test set. (Download the PascalVOC2007 test set using `wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar` and run `tar -xvf VOCtest_06-Nov-2007.tar` in the root directory of the repository.
### Performance 


| Category  | Clean Image (mAP) | Noisy Image (mAP) | Denoised Image (mAP) |
| ------------- | ------------- | ------------- | ------------- |
| Training  |  88.19% | 52.73% | 73.78% | 
| Evaluation  | 77.43% | 46.47% | 61.84% |

### Experiments
- **Training & Optimization Experiments** (Plots for all these experiments can be found inside `optimization_experiments/` folder). The experiemt was run over multiple optimizers as shown below. 
  - **SSD_train.ipynb** - Runs the training using SGD Optimizer.
  - **SSD_train_Adam.ipynb** - Runs the training using Adam Optimizer.
  - **SSD_train_RMSProp.ipynb** - Runs the training using RMSProp Optimizer.
  - **SSD_train_LearningRate.ipynb** - Runs the training using a range of *learning rates*. Used for hyperparameter tuning.
  - **SSD_train_Momentum.ipynb** - Runs the training using a range of *momentum* values. Used for hyperparameter tuning.

### Directory structure
- data/ -
  - init.py - contains instances:
    - function detection_collate - stack images in 0th dimension and list of tensors with annotations for image and return in tuple format, given tuple of tensor images and list of annotations
    - function base_transform - resize and mean-normalize image
    - class BaseTransform - call base_transform(image) iteratively
  - config.py - configures VOC dataset with source directory, mean values, color ranges and SSD parameters
  - voc0712.py - configures VOC dataset with labels considered, and contains instances:
    - class VOCAnnotationTransform - store dictionaries of classname:index mappings, with an option to discard difficult instances
    - class VOCDetection - update and store annotation based on input image, with functions to get item, pull item, image, annotation and tensor
- demos/ - demo gifs to show performance of SSD on noisy, clean and denoised video streams (source files for the .gifs shown above)
- \*_experiments/ - experiments folders for denoising, optimization and video performance evaluation
  - .ipynb_checkpoints/ - checkpoints folder for modular running of python notebooks
  - \*.ipynb - jupyter notebooks to visualize descent of loss, other evaluation metrics
  - \*.jpeg - plots of loss functions in different scenarios
  - pickles/ - pickle files for easy storing of data during cross validation (different learning rates, momentums etc.)
  - NOISE_PARAMS.pkl - Pickle file for noise parameters
  - nntools.py - class script for base classes to implement neural nets, evaluate performance, specify metrics etc.   
- architectures/-
    - Network changes to the baseline model
- layers/ 
  - functions/ - 
    - init.py - import all files in pwd
    - detection.py - contains instances:
      - class Detect - enable decoding of location predictions of bboxes and apply NMS based on confidence values and threshold; restrict to tok_k output predictions to reduce noise in results quality (not actual image noise)
        - function init - allocate memory and initialize
        - function forward - forward propagation to update layers given input location prediction, confidence and prior data from their respective layers
    - prior_box.py - contains instances:
      - class PriorBox - collate and store priorbox coordinates in center-offset form and tie it to each source feature map
        - function init - allocate memory and initialize
        - forward - forward propagation through priorbox layers
  - modules/ -
    - init.py - import all files in pwd
    - l2norm.py - contains instances:
      - class L2Norm - calculate L2 norm and normalize
        - function init - allocate memory and initialize
        - forward - compute the norm and return
    - multibox_loss.py - contains instances:
      - class MultiBoxLoss - compute targets for confidence and localization and apply HNM; using a loss function that is weighted between the cross entropy loss and a smooth L1 loss (weights were found during cross validation)
        - function init - allocate memory and initialize
        - function forward - forward propagate through multibox layers, given tuple of location and confidence predictions, prior box values and ground truth boxes and labels in tensor format
  - init.py - import all files in pwd
  - box_utils.py - contains instances:
    - function point_form - convert prior box values from center-size format for easy comparison to point form ground truth data
    - function center_size - convert prior box values to center-size format for easy comparison to center-size ground truth data
    - function intersect - compute area of intersection between two given boxes
    - function jaccard - compute jaccard overlap or intersection over union of two boxes
    - function match - match prior box with ground truth box (for all boxes) based on highest jaccard overlap, encode in required format (point-form or center-size), and return matching indices for the given confidence and location predictions
    - function encode - encode variances from prior box layers into ground truth boxes
    - function decode - decode locations from priors and locations and return bbox predictions
    - function log_sum_exp - compute log of sum of exponent of difference between current tensor and maximum value of tensor, for unaveraged confidence loss
    - function nms - compute non-maximum suppression to avoid too many overlapping bboxes that highlight nearly the same area
- utils/ -

  - init.py - import all in pwd
  - augmentations.py - contains instances:
    - function intersect - return intersection of two given bounding boxes
    - function jaccard_numpy - return IoU or jaccard overlap of two given bounding boxes
    - class Compose - definitions of different transforms to perform
    - class Lambda - applies a lambda as a transform
    - class ConvertFromInts - convert object from integers
    - class SubtractMeans - subtract mean of image from passed image for normalization
    - class ToAbsoluteCoords - convert lengths (widths, heights) to absolute coordinates
    - class ToPercentCoords - convert coordinates to percentage values of image height and width
    - class Resize - resize image
    - class RandomSaturation - randomly saturate an image
    - class RandomHue - add a random hue to an image
    - class RandomLightingNoise - add random lighting noise to an image
    - class ConvertColor - convert colorspace from BGR to HSV or vice versa
    - class RandomContrast - add random contrast to an image
    - class RandomBrightness - add random brightness to an image
    - class ToCV2Image - shift image to CPU
    - class ToTensor - shift image to GPU
    - class RandomSampleCrop - randomly crop an image and return cropped image, adjusted bounding boxes and new class labels
    - class Expand - expand an image through zero padding and mean-filling, and return along with adjusted bounding boxes and new class labels
    - class RandomMirror - randomly choose to mirror an image
    - class SwapChannels - Transform image by swapping channels in the specified order
    - class PhotometricDistort - apply random brightness and lighting noise, and randomly distort images
    - class SSDAugmentation - itemize all the above transformation functions on every image iteratively
- weights/ - \*.pth files containing pretrained weights of SSD for the VOC 2012 dataset
- requirements.txt - package and module requirements for running the project
- opts.py - Contains default variables, parameters, and paths to run experiments.
- denNet.pt - Trained DUDnCNN net for image denoising.

### References 

- [Project Statement for Object-Detection](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)
- Inspired by https://github.com/amdegroot/ssd.pytorch for SSD Implementation.

Regarding the Inception Implementation, we referred to:
- [INCEPTION SINGLE SHOT MULTIBOX DETECTOR FOR OBJECT DETECTION ](https://sci-hub.tw/10.1109/icmew.2017.8026312)

## Contributers - 

- Vedavyas Poturu
- Sanika Patange
- Payam Khorramshahi
- Shivani Athavale
