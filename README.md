# VisualVectorizingNetwork
1;95;0cModels for converting image-like inputs into hierarchical graph-like scene representations

Installation Instructions:

0. Create a python2 virtual environment (highly recommended)
    ```
    cd [WORKING_DIR]
    virtualenv env -p python
    source env/bin/activate
    ```
1. Clone and install the `vvn` package:
    ```
    git clone git@github.com:neuroailab/PSGNets.git
    cd PSGNets && pip install -e .
    ```
2. Check installation by training a ResNet18 on ImageNet categorization [Note: this uses the TFutils training script]:
    ```
    cd psgnets/trainval
    python train_psgnet_tfutils.py --gpus [X] --config_path ./training_configs/resnet_config.py --exp_id [MY_EXP_ID] --batch_size 256 --save_dir ./training_configs/tfutils_params --port [PORT_TO_MONGODB] [--data_dir [PATH_TO_IMAGENET]]
    ```
  where [X] is the number of a free GPU on the node you're on, [MY_EXP_ID] is a name for your training experiment, [PORT_TO_MONGODB] is a valid locahost port hosting a mongodb, and [PATH_TO_IMAGENET] is an optional path where Imagenet data are stored. If you don't pass an argument to --data_dir, the model will train from a default directory specified in `psgnets/data/imagenet_data.py`.
