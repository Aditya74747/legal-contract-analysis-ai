# Models

The fine-tuned model (~250 MB) is too large for git, so it is not stored in this repository.

**Option 1: download it.** Get `best_model.zip` from the
[Releases page](https://github.com/Aditya74747/legal-contract-analysis-ai/releases/latest)
and extract it so that the files end up in:

```
models/finetuned/best_model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json ...
```

**Option 2: train it yourself.** Run `notebooks/01_prepare_data.ipynb` and then
`notebooks/02_finetune.ipynb`. The model is saved to `models/finetuned/best_model/`.
