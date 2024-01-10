# Federated Semi-Supervised Learning with Truncated Gaussian Aggregation

> Currently working on PyTorch version 

Due to the high cost of labeling and the high requirements of annotation professionalism, large amounts of data lack annotation. As a solution to the problem of partially labeled data in federated learning, federated semi-supervised learning has emerged. To take advantage of the large volume of unlabeled data to improve model performance, we propose a semi-supervised federated learning method based on truncated Gaussian aggregation, named TGAFed, which focuses on the case where each client has access to both labeled and unlabeled data in the federated semi-supervised learning. To optimize the filtering of pseudo-labels, the unlabeled samples will be weighted according to the truncated Gaussian distribution fitted by the model prediction probability of the unlabeled samples. Furthermore, a new inter-client consistency loss is generated, incorporating the truncated Gaussian distribution, to enhance the utilization rate of pseudo-labels. Then, the server performs mean aggregation based on local quantity-quality factors, while global quantity-quality factors assist clients in their local updates through exponential moving averages, gradually improving the performance of the global model. We empirically validate our method on both IID and Non-IID scenarios. The results demonstrate that our method surpasses the base model, FedMatch, on both the Cifar10 and Cifar100 datasets.

The main contributions of this work are as follows:

* We introduce a pseudo-label weight prediction method that improves client pseudo-label utilization of unlabeled samples by fitting a truncated Gaussian threshold distribution to weight the inter-client consistency loss of unlabeled samples.
* The model parameters and the local quantity-quality factors are mean-aggregated, and the generated global quantity-quality factors are used to assist the client in parameter updating utilizing exponential moving averaging, which prevents the degradation of global model performance due to the forgetfulness of pseudo-labeling knowledge by the local model. 
* The TGAFed method outperforms the base model FedMatch in both IID and Non-IID settings on the benchmark datasets Cifar10 and Cifar100.
	

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
The following tasks will be generated from `CIFAR-10`.

* **`lc-biid-c10`**: `bath-iid` task in `labels-at-client` scenario
* **`lc-bimb-c10`**: `bath-non-iid` task in `labels-at-client` scenario


## Run Experiments
To reproduce experiments, execute `train-xxx.sh` files in `scripts` folder, or you may also run the following comamnd line directly:

```bash
python3 ../main.py  --gpu 0,1,2,3,4 \
            --work-type train \
            --model tgafed \
            --task lc-biid-c10 \
            --frac-client 0.05 \
```

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results, such as local & global performance and communication costs (S2C and C2S), and the experimental setups, such as learning rate, batch-size, number of rounds, etc. The log files will be updated for every comunication rounds. 

## Citations
```

```
