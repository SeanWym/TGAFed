## Environmental Setup

Please install packages from `requirements.txt` after creating your own environment with `python 3.8.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Data Generation
Please see `config.py` to set your custom path for both `datasets` and `output files`.
```python
args.dataset_path = '/path/to/data/'  # for datasets
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Run below script to generate datasets
```bash
$ cd scripts
$ sh gen-data.sh
```

## Run Experiments
To reproduce experiments, execute `train-xxx.sh` files in `scripts` folder, or you may also run the following comamnd line directly:

```bash
python3 ../main.py  --gpu 0,1,2,3,4 \
            --work-type train \
            --model fedmatch \
            --task lc-biid-c10 \
            --frac-client 0.05 \
```
Please replace an argument for **`--task`** with one of `lc-biid-c10`, `lc-bimb-c10`, `ls-biid-c10`, and `ls-bimb-c10`. For the other options (i.e. hyper-parameters, batch-size, number of rounds, etc.), please refer to `config.py` file at the project root folder.

> Note: while training, 100 clients are logically swiched across the physical gpus given by `--gpu` options (5 gpus in the above example). 

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results.

