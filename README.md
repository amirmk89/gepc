# Graph Embedded Pose Clustering for Anomaly Detection
This is the code for ["Graph Embedded Pose Clustering for Anomaly Detection".](https://arxiv.org/abs/1912.11850)

## Prerequisites
- Pytoch 1.2.0
- Faiss
- Numpy
- SciPy
- Sklearn
- Tqdm
- Lmdb
- PyArrow
- Pillow 
 
Detailed dependencies are provided in the 'environment.txt' file. 

## Getting Started
```
git clone https://github.com/amirmk89/gepc
cd data
./unpack.sh  # Unpack pose data, GT, and patch data (sample only)

# Conda environment setup
cd ..
conda env create -n gepc -f environment.txt
pip install lmdb
```

## Directory Structure
```
.
├── models       -- Including graph definitions and convolution operators
├── utils
├── data            -- Configurable and may be moved
├── LICENSE  
├── README.md
├── environment.txt      -- For creating the conda environment
└── stc_train_eval.py     -- Main file for training / inference
```

### Data Directory
The data directory holds pose graphs and ground truth vectors for the dataset.
A path for the directory may be configured using the --data_dir argument or by 
creating symlinks inside the project's data directory.

```
.
├── pose  -- Extracted temporal pose graphs
│   └─ training  
│   └─ testing
│   └─ unpack.sh
│
├── training  
│   └─ videos
│
├── testing    
│   └─ frames
│   └─ test_frame_mask  -- GT vectors
│
└── gen_patch_db.py  -- For serializing dataset into LMDB
```



### Patch Training
For running the patch based variant, it is needed to download the [ShanghaiTech Campus dataset](https://svip-lab.github.io/dataset/campus_dataset.html), and extract it according
to the specified data directory structure. The training data is provided in video clips while the test data is provided
as individual frames. For training it is required to split the input training videos to individual frames, e.g. using FFMPEG.

The model supports patch training using both frame .jpeg files and using a serialized patch file in lmdb format. 
For performance reasons, the use of patch DB files is highly recommended. 
A script for creating the serialized patch DB files is provided in the data directory. 


## Training Script

### Pose
Assuming data is located in the expected path (or --data_dir argument is used), pose based training is run
using:
```
python stc_train_eval.py
```
A shorter run with a fraction of the data can be done the verify everything is properly set using the --debug flag.

### Patch Training
Use the following command (assuming a pre-extracted patch DB):
```
python stc_train_eval.py --patch_features --patch_db -ae_b 256 -dcec_b 256 -res_b 128 
```


## Citation
If you find this useful, please cite this work as follows:

    @misc{markovitz2019graph,
        title={Graph Embedded Pose Clustering for Anomaly Detection},
        author={Amir Markovitz and Gilad Sharir and Itamar Friedman and Lihi Zelnik-Manor and Shai Avidan},
        year={2019},
        eprint={1912.11850},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

