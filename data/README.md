# Data

This project uses the **CUAD v1** contract dataset from [The Atticus Project](https://www.atticusprojectai.org/cuad)
(licensed under **CC BY 4.0**). It is not included in this repository because of its size.

1. Download the dataset, e.g. from Kaggle:
   https://www.kaggle.com/datasets/konradb/atticus-open-contract-dataset-aok-beta
2. Extract it so that the folder looks like this:

```
data/raw/
├── CUAD_v1.json            SQuAD-format QA annotations (main source)
├── master_clauses.csv      One row per contract, one column per clause type
├── label_group_xlsx/       28 Excel files, one per clause category
├── full_contract_txt/      510 contracts as plain text
└── full_contract_pdf/      The same contracts as PDFs (not used for training)
```

3. Run `notebooks/01_prepare_data.ipynb`. It writes the cleaned dataset to `data/processed/`.
