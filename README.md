# CRL
Implementation of continual learning research.

## Dependencies

Use anaconda to create python environment:

> conda create --name yourname python=3.8 \
> conda activate yourname

Install Pytorch (suggestions>=1.7) and related environmental dependencies:

> pip install -r requirements.txt

Pre-trained BERT weights:
* Download *bert-base-uncased* into the *datasets/* directory [[google drive]](https://drive.google.com/drive/folders/1BGNdXrxy6W_sWaI9DasykTj36sMOoOGK).

### Run the Code

> python run_continual.py --dataname FewRel

### some example for running:
```
job`s:
- name: $EXP_NAME-TOTAL_RANDOM
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 1 --change_query 0 --must_every_class 0
- name: $EXP_NAME-retrieve_topk
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 0 --change_query 0 --must_every_class 0
- name: $EXP_NAME-retrieve_topk-high_lr
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 0 --change_query 0 --must_every_class 0 --learning_rate 2e-5
- name: $EXP_NAME-0.5random-0.5topk
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 0.5 --change_query 0 --must_every_class 0 
- name: $EXP_NAME-retrieve_every_class-high_lr
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 0 --change_query 0 --learning_rate 2e-5 --must_every_class 1
- name: $EXP_NAME-retrieve_every_class
  sku: G1
  command:
  - python -u run_continual.py --output_path $$AMLT_OUTPUT_DIR --exp_name $EXP_NAME --bert_path bert-base-uncased --num_protos 20 --step1_epochs 10 --step2_epochs 10 --total_round 5 --retrieve_random_ratio 0 --change_query 0 --must_every_class 1
```

### Explanation of some parameters:
```
--output_path: output path 
--exp_name: experiment name 
--bert_path: bert path from above link
--num_protos: number of save for each class
--step1_epochs: epoch of new data training 
--step2_epochs: epoch of replaying
--total_round: Number of repetitions, this will change task order
--retrieve_random_ratio: in retrieval, ratio of random. If set to 1, then random will be done.
--change_query: change query every time.
--must_every_class: If set to 1, then retrieve from every class instead of retrieving by score.
```


### Some other version:

Statistic Memory Version:
tag is: separate prototype and training data
url is: https://github.com/NONOThingC/NewTest/tree/7b1e9a8aff26fcb6bc67c70202cb58e47d6700ec

Prototypical Version:
tag is: prototypical training
url is: https://github.com/NONOThingC/NewTest/commit/9210dd35fa326be7309d4d03aaecec75b3a8f6c7

