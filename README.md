# Kaggle: LLM Prompt Recovery - 3rd Place Solution
This repository contain parts of the code to generate the 3rd place solution to the Kaggle's LLM Prompt Recovery competition.
You can find the details for the solution in the [write-up post](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494621)

This repo contains the training scripts for the gate, clustering and tags models.

The two dataset files `dataset.csv` (for gate and cluster models) and the `dataset_tags.csv` (for the tag model) are also available [here](https://www.kaggle.com/datasets/arc144/llm-prompt-recovery-gate-cluster-tags-datasets) as a Kaggle dataset.

Before running the training scripts just make sure to create a new environment from the `requirements.txt` file, [download the datasets from the provided link](https://www.kaggle.com/datasets/arc144/llm-prompt-recovery-gate-cluster-tags-datasets) and unzip them here in the root directory.