# TrkGNN


## Getting started (bl-0.inpac.sjtu.edu)
```bash
# go to the working directory and
# clone repository from gitlab
git clone git@gitlab.com:yulei_zhang/trkgnn.git

# initiate the environment
source /sw/anaconda/3.7-2020.02/thisconda.sh 
conda activate /lustre/collider/zhangyulei/DeepLearning/env

# adding project path to PYTHONPATH
GNNPATH=<path-to-trkgnn>
export PYTHONPATH="${PYTHONPATH}:${GNNPATH}"
```

## How to run? 
```bash
python3 ${GNNPATH}/run.py --help

# usage: GNN [-h] {test,DDP} ...

# GNN Tracking Toolkit

# optional arguments:
#  -h, --help  show this help message and exit

# modules:
#   {test,DDP}  sub-command help
#     test      quick test using small dataset
#     DDP       Distributed Data Parallel Training


python3 ${GNNPATH}/run.py DDP --help

# usage: GNN DDP [-h] [-w WORLD_SIZE] [-v] config

# positional arguments:
#   config                the config file

# optional arguments:
#   -h, --help            show this help message and exit
#   -w WORLD_SIZE, --world_size WORLD_SIZE
#   -v, --verbose
```

<b>Test is only for development usage. DDP is the main training module.</b>
World size `-w` is the number of GPUs to use (only use 1 in bl-0 unless you've requested multiple GPUs in condor). 
`-v` is for verbose mode.
Config is the path to configuration file, which is a yaml format. An example is shown below:
```yaml
# training output directory
output_dir: ./output
# random number seed
rndm: 1

data:
  #--------------------------------
  # There are two ways to read data:
  # 1. Read from ROOT (default)
  # 2. Read from graph
  #    -- Graph is built from script: `extra_script/graph_to_disk.py`
  #    -- One Graph file is a list of graphs to be one chunk
  #--------------------------------
  # --> Read from ROOT (disable if read from graph)
  # input directory for data
  input_dir: "/Users/avencast/CLionProjects/darkshine-simulation/workspace/Tracker_GNN*.root"
  tree_name: 'dp'
  # how many total events to train
  global_stop: 1000000
  # how many events loaded per training period (Not total)
  chunk_size: "50 MB"

  # --> Read from graph
  read_from_graph: true
  # input directory for graph
  input_graph_dir: "/Users/avencast/CLionProjects/darkshine-simulation/workspace/test_output/"
  # how many files to read per training period (Not total) 
  # (set to -1 to read all files)
  global_stop_graph_file: -1

  # --> General Data Loader Settings
  # which collection to use
  collection: 'TagTrk1'
  batch_size: 1
  # number of works using CPU (0 disable multi-threading)
  n_workers: 0
  # initial incident energy (normalization constant)
  E0: 2900
  # move large graph to only validation stage
  min_graph_size: 1 # MB
  
  # This is mainly for application stage (automatically true)
  #   -- it's more or less determined in the training stage and graph building stage
  #   -- If input graph or trained model has 3 features, then it's trained without BField
  #   -- If input graph or trained model has 6 features, then it's trained with BField
  graph_with_BField: true
  

# make sure to choose the correct model if to predict momentum
momentum_predict: true
model:
  # model to predict momentum
  name: mpnn_p
  # data input dimension (x,y,z) = 3
  input_dim: 3
  # network settings
  n_edge_layers: 4
  n_node_layers: 4
  hidden_dim: 64
  n_graph_iters: 8
  layer_norm: true

# loss function
loss_func: binary_cross_entropy_with_logits

optimizer:
  name: Adam
  learning_rate: 0.001
  weight_decay: 1.e-5
  # decay shcedule for optimizer
  lr_decay_schedule:
    - { start_epoch: 36, end_epoch: 64, factor: 0.1 }
    - { start_epoch: 64, end_epoch: 82, factor: 0.01 }
    - { start_epoch: 82, end_epoch: 100, factor: 0.01 }

training:
  # training epochs
  n_epochs: 90
  # total training epochs
  n_total_epochs: 100
```

- Since the dataset is large (impossible to load all events in RAM in one time), we load data in chunks. `chunk_size` is the size of each chunk.
- For each chunk, we load `batch_size` events to GPU to train.
- `global_stop` is the total number of events to train.
- `E0` is the normalization constant for incident energy. The truth/predicted momentum is normalized by `E0`.
- `input_dir` is the example input directory. You can change it to your own directory if you have any new root files.

### How to run on bl-0

Running GPU jobs on bl-0 is a little bit tricky. You need to request a GPU node first, then submit a job to the node. 
2 files are needed for submitting a job: `run.sh` and `submit.condor`.
```bash
# run.sh
#!/bin/bash

source /sw/anaconda/3.7-2020.02/thisconda.sh 
conda activate /lustre/collider/zhangyulei/DeepLearning/env

GNNPATH="/lustre/collider/zhangyulei/DeepLearning/Tracking/trkgnn"

export PYTHONPATH="${PYTHONPATH}:${GNNPATH}"
python3 ${GNNPATH}/run.py DDP mpnn.yaml -w 1
```
```bash
# submit.condor
Universe   = vanilla
Executable = /lustre/collider/zhangyulei/DeepLearning/Tracking/workspace/run.sh
# Arguments  =
Log        = log/track.log
Output     = log/track.out
Error      = log/track.err
request_GPUs = 1
request_CPUs = 1
#rank = (OpSysName == "CentOS")
#requirements = (machine == "bl-hd-1.phy.sjtulocal")

Queue
```
Make sure you change the executable path in `submit.condor` to your own path. 
Make sure you have log directory in your working directory.
Then submit the job using `condor_submit submit.condor`. 
You can query the job by `condor_q` and check the log file by `tail -f log/track.log`.

Example working directory:
```
/lustre/collider/zhangyulei/DeepLearning/Tracking/workspace
```

## How to check the training progress?

The easiest way is to check the log file. It should be located in `./output` folder. For each GPU, there is a log file and a CSV file.
Log file records the details of training, and CSV file records the numbers for plotting.

```bash
# log file
2023-05-16 21:23:42,392 INFO Added key: store_based_barrier_key:1 to store for rank: 0
2023-05-16 21:23:42,392 INFO Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
2023-05-16 21:23:44,551 INFO Epoch 0
2023-05-16 21:24:13,031 INFO Reducer buckets have been rebuilt in this iteration.
2023-05-16 21:25:12,875 INFO   Training loss: 0.927
2023-05-16 21:25:22,696 INFO   Validation loss: 0.606 acc: 0.440
2023-05-16 21:25:23,013 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
2023-05-16 21:26:45,355 INFO   Training loss: 0.594
2023-05-16 21:26:55,060 INFO   Validation loss: 0.576 acc: 0.553
2023-05-16 21:26:55,068 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
2023-05-16 21:28:16,969 INFO   Training loss: 0.515
2023-05-16 21:28:26,738 INFO   Validation loss: 0.328 acc: 0.739
2023-05-16 21:28:26,745 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
2023-05-16 21:29:48,770 INFO   Training loss: 0.176
2023-05-16 21:29:58,498 INFO   Validation loss: 0.088 acc: 0.950
2023-05-16 21:29:58,505 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
2023-05-16 21:31:20,714 INFO   Training loss: 0.068
2023-05-16 21:31:30,441 INFO   Validation loss: 0.055 acc: 0.969
2023-05-16 21:31:30,448 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
2023-05-16 21:31:40,527 INFO [Rank 0]: Write checkpoint 0 to model_checkpoint_000.pth.tar
2023-05-16 21:31:40,528 INFO Epoch 1
2023-05-16 21:33:04,103 INFO   Training loss: 0.046
2023-05-16 21:33:13,868 INFO   Validation loss: 0.042 acc: 0.979
2023-05-16 21:33:13,875 INFO [Rank 0]: Write summary to ./output/summaries_0.csv
```

| lr   | train_loss |      l1      |      l2      |   grad_norm  | train_batches | itr | epoch | valid_loss | valid_acc | valid_TP | valid_FP | valid_TN | valid_FN | valid_batches | valid_sum_total | valid_dp_mean | valid_dp_std |
|------|------------|--------------|--------------|--------------|---------------|-----|-------|------------|-----------|----------|----------|----------|----------|---------------|-----------------|---------------|--------------|
| 0.001|  0.926528  | 3377.137857  | 26.725477    | 4.007107     | 133           | 0   | 0     | 0.605998   | 0.440378  | 322884   | 7056094  | 5319162  | 113678   | 57            | 12811818        | 0.103146      | 3.44327      |
| 0.001|  0.594081  | 3399.439001  | 26.550938    | 10.386797    | 133           | 1   | 0     | 0.575632   | 0.552759  | 337311   | 5592230  | 6696912  | 99207    | 57            | 12725660        | 0.097205      | 1.82003      |
| 0.001|  0.514561  | 3512.446358  | 26.737954    | 31.345152    | 133           | 2   | 0     | 0.327967   | 0.738962  | 435977   | 3340114  | 9021071  | 582      | 57            | 12797744        | -0.215187     | 1.09541      |
| 0.001|  0.176163  | 3584.008306  | 27.005496    | 6.185369     | 133           | 3   | 0     | 0.087850   | 0.950204  | 435869   | 631843   | 11633319 | 645      | 57            | 12701676        | -0.0602811    | 1.49604      |
| 0.001|  0.068454  | 3654.412476  | 27.230886    | 17.664845    | 133           | 4   | 0     | 0.055039   | 0.968504  | 435960   | 400069   | 11883665 | 566      | 57            | 12720260        | 0.125798      | 4.20775      |
| 0.001|  0.045525  | 3666.339746  | 27.285557    | 6.994221     | 133           | 0   | 1     | 0.041694   | 0.979115  | 435851   | 266866   | 12108390 | 711      | 57            | 12811818        | 0.0842447     | 1.92401      |

### Visualization
There are three jupyter notebooks in `visualization` folder in the repository.

- [evaluation_graph.ipynb](visualization%2Fevaluation_graph.ipynb) is for evaluating the training progress. 
You can download a small set of data and model to local and run the notebook to see the evaluation.
- [monitor_ihep.ipynb](visualization%2Fmonitor_ihep.ipynb) is for monitoring the training progress on cluster, IHEP for example. 
- [visual_graph.ipynb](visualization%2Fvisual_graph.ipynb) is for visualizing the graph in data file.

## Full workflow

This is an example full workflow of the project.
1. Run DSS framework to generate `Tracker_GNN.root` files (Generated by `DAna:GNN_DataExporter`)
   - The beam energy (normally should be 8 GeV) &rarr; `E0` as energy normalization factor
   - Whether to use const b field (normally should be -1.5 T along y-axis)
     - If using const b field, it's better not to include the b field in the input features. **Dimension of input features: 3`(x,y,z)`**.
     - If using non-uniform b field, adding the b field in the input features. **Dimension of input features: 6`(x,y,z,Bx,By,Bz)`**.
2. (_**Optinal**_) Generate Graphs for training. (accelerate the training process)
```bash
python3 trkgnn/extra_script/graph_to_disk.py Tracker_GNN.root -o output -c 30MB -b
```
3. Run training 
```bash
python3 trkgnn/run.py DDP config.yaml -w 1
```
  - If with b-field, set `graph_with_BField: true` and `input_dim: 6` in `config.yaml`.
  - If train with graph from step 2, set `read_from_graph: true` in `config.yaml`.
  - Models and plots should be saved in `output_dir` in the `config.yaml`.
4. Run application
```bash
python3 trkgnn/run.py apply apply Tracker_GNN.root -m output/model.checkpoints/model_checkpoint_011.pth.tar -c config.yaml -o output
```
5. Repeat step 3 and 4 for both tagging region and recoil region. Save the final evaluated results in `tag.root` and `rec.root`.
6. Merged the GNN results with the original `dp_ana.root` file
```bash
 root -l -b -q 'trkgnn/extra_script/merge_to_DAna.cpp("tag.root", "rec.root", "dp_ana.root")'
```
7. Do normal analysis using `dp_ana.root` file.

## Environments
```bash
# create conda environment
#conda create -p /lustre/collider/zhangyulei/DeepLearning/ml python=3.9

# nivida cuda toolkit
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolki

# torch
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg pytorch-scatter -c pyg
pip install torchinfo  

# general
conda install pandas pyyaml
pip install uproot seaborn wandb

# plots
conda install -c plotly plotly
 
 
```

