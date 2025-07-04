# training-disk-experiment

## Environment
Assume you have a Ray cluster with 3 workers nodes, and you have already deployed Kuberay operator in the cluster.

## Deploy the Ray cluster

```bash
kubectl apply -f ray-cluster.yaml
```

## Run the experiment with COCO

```bash
ray job submit --runtime-env-json='{"working_dir": ".", "pip": ["torch", "torchvision", "pandas", "pycocotools"]}' -- python coco_experiment.py
```