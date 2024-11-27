

Clone `Few_Shot_transformation_and_sampling` from the paper [Revisiting Few-shot Relation Classification: Evaluation Data and Classification Schemes](https://arxiv.org/abs/2104.08481):

```bash
git clone https://github.com/ofersabo/Few_Shot_transformation_and_sampling.git
cd Few_Shot_transformation_and_sampling
```

Download TACRED:

```bash
wget https://nlp.stanford.edu/~manning/xyzzy/tacred-with-json.zip
unzip tacred-with-json.zip 
```

```bash
mkdir -p tacred/data/instances_per_relation
python convert_dataset_to_list_by_relation.py --dataset tacred/data/json/train.json --output_file tacred/data/instances_per_relation/TACRED_train.json
python convert_dataset_to_list_by_relation.py --dataset tacred/data/json/dev.json --output_file tacred/data/instances_per_relation/TACRED_dev.json
python convert_dataset_to_list_by_relation.py --dataset tacred/data/json/test.json --output_file tacred/data/instances_per_relation/TACRED_test.json
```


```bash
python data_transformation.py --train_data tacred/data/instances_per_relation/TACRED_train.json --dev_data tacred/data/instances_per_relation/TACRED_dev.json --test_data tacred/data/instances_per_relation/TACRED_test.json --fixed_categories_split categories_split.json --test_size 10 --output_dir ./data_few_shot
```



