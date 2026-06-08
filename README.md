









## Train

```
py -3 train_model.py --train-dir dataset/train --dev-dir dataset/dev-0 --model-dir models
```

## Majority

```bash
py -3 predict.py --input dataset/test-A/in.tsv --model-dir models --output predictions_testA.tsv --method ml
```

```bash
py -3 predict.py --input dataset/test-B/in.tsv --model-dir models --output predictions_testB.tsv --method ml
```

## Majority

```bash
py -3 predict.py --input dataset/test-A/in.tsv --model-dir models --output predictions_testA_majority.tsv --method majority
```
```bash
py -3 predict.py --input dataset/test-B/in.tsv --model-dir models --output predictions_testB_majority.tsv --method majority
```

##
