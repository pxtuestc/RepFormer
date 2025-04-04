# Introduction  
This repository contains the PyTorch implementation code for **RepFormer**.  

Before execution, you need to:  
1. Download the **RML2016.10A** and **RML2018.01A** datasets.  
2. Place them in the `<datasets>` directory.  
3. Modify the dataset-related parameters in the script `<main_repformer.py>` for training and validation.  

We provide a pre-trained RepFormer model (trained on **RML2016.10A**) in the `<checkpoints>` directory for validation purposes.  

To run validation, simply execute the `<main_repformer.py>` script. It will evaluate both the **non-reparameterized** and **reparameterized** models on the test set and report their model architectures and accuracies.  
