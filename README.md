# "Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness" Project

## Overview
This project focuses on our proposed knowledge unlearning framework for LLMs. This codebase provides the official implementation of our framework introduced in the paper.

## Environment Setup

### Method 1: Using conda
``` sh
conda env create -f environment.yml
conda activate knowledge_unlearning
```

### Method 2: Using pip
``` sh
conda create -n knowledge_unlearning python=3.10
conda activate knowledge_unlearning
pip install -r requirements.txt
```
## Project Structure
```
knowledge_unlearning/
│── src/
│   ├── data/
│   │   ├── knowledge_graph.py  # Handles knowledge graph loading
│   │   ├── text_conversion.py  # Converts KGs to text-based samples            
│   │   ├── yago3-10/ # Select yago3-10 as the knowledge graph
│   ├── models/
│   │   ├── model_loader.py  # Load Model, Tokenizer, and Model Config
│   ├── evaluation/
│   │   ├── evaluator.py  # Query performance of the target LLM and construct subgraph
│   │   ├── llm_evaluator.py  # Evaluate subgraph and target unlearning triples with powerful LLMs
│   ├── main.py  # Main entry point for the project
│   ├── utils/
│   │   ├── utils.py
│   ├── unlearning/
│   │   ├── unlearn.py
│   │   ├── methods/
│   │   │   ├── gradient_ascent.py
│   │   │   ├── random_label.py
│   │   │   ├── negative_preference_optimization.py
│   │   │   ├── ascent_plus_descent.py
│   │   │   ├── scrub.py
│   ├── config/
│   │   ├── config.yml
│   │   ├── config_judge.yml
│   │   ├── config_subgraph.yml
│   │   ├── config_threshold.yml
│   │   ├── config_unlearn.yml
│   │   ├── config_utility.yml
│   ├── README.md
```

## Usage

### Unlearn: Load Knowledge Graphs, Unlearn Target Triples over the Target LLM
```sh
nohup deepspeed --include localhost:1,2 --master_port=33333 src/main_finetune.py --config config/config_unlearn.yml --unlearn > output.log 2>&1 &
```


#### Important Parameters in `config_unlearn.yml`

- `huggingface`  
  - `hf_token`: The token associated with your Hugging Face account

- `model`  
  - `name`: Name of the unlearning model

- `unlearn`  
  - `use_QA_unlearning`: Set to `true` to use QA unlearning, `false` to use Sentence unlearning  
  - `peft_config`: `"LoRA"` indicates using LoRA-based unlearning; any other value indicates full-parameter unlearning  
  - `method`: Unlearning method to be applied

- `output`  
  - `unlearn_model_dir`: Path to save the checkpoint



### Construct Supporting Subgraph:
```sh
nohup python src/main_finetune.py --config config/config_subgraph.yml --subgraph > output.log 2>&1 &
```

#### Important Parameters in `config_subgraph.yml`

- `huggingface`  
  - `hf_token`: The token associated with your Hugging Face account

- `model`  
  - `name`: Name of the unlearning model  
  - `merge_lora`: Set to `true` if the unlearned model was unlearned using LoRA; set to `false` if it was unlearned using full-parameter unlearning  
  - `save_merged_model_path`: If the unlearned model is trained using LoRA unlearning, this specifies the path to save the merged model after combining the LoRA adapter with the original model  
  - `checkpoint_path`: If the unlearned model is trained using LoRA unlearning, this is the path where the LoRA adapter is saved. If the model is trained with full-parameter unlearning, this is the path where the model parameters are saved  

- `subgraph`  
  - `k_hop`: Number of hops for the Supporting Subgraph  
  - `unlearn_method`: Unlearning method to be applied  
  - `eval_unlearn_triples`: Should be set to `true` when constructing the Supporting Subgraph  
  - `score_unlearn_triples`: Should be set to `false` when constructing the Supporting Subgraph  

- `output`  
  - `subgraph_dir`: Directory to save the extracted subgraph



### Evaluate Unlearn Triples:
```sh
nohup python src/main_finetune.py --config config/config_subgraph.yml --subgraph > output.log 2>&1 &
```

Similar to **Construct Supporting Subgraph**, but with the following settings:

- `eval_unlearn_triples`: set to `false`
- `score_unlearn_triples`: set to `true`



### Evaluate Utility Triples:
```sh
nohup python src/main_finetune.py --config config/config_utility.yml --utility > output.log 2>&1 &
```

#### Important Parameters in `config_utility.yml`

**utility**  
- `run_normal`: Specify whether to perform normal utility evaluation.

**output**  
- `score_dir`: Directory where the results will be saved.


### Evaluation with Powerful LLMs:
```sh
nohup python src/main_finetune.py --config config/config_judge.yml --judge > output.log 2>&1 &
```

#### Important Parameters in `config_judger.yml`

**input**  
- `judge_superficial`: Specify whether to score only the unlearn triples.  
- `judge_superficial_path`: Path to the results saved after Evaluate Unlearn Triples.  
- `judge_subgraph_path`: Path to the results saved after Construct Supporting Subgraph.  

**output**  
- `output_superficial_path`: Path to save the scoring results for unlearn triples.  
- `evaluation_output_file`: Path to save the scoring results for supporting subgraphs.


## Supported Models
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct

## Supported Unlearning Methods
- gradient_ascent
- random_label
- npo
- ascent_plus_descent
- scrub
