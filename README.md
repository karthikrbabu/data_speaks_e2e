<!-- #region -->
# data_speaks_e2e

This repository is made to support our research paper ```Data Speaks, E2E Data-to-Text with T5```

**Authors: Shyamkarthik Rameshbabu, Praveen Kasireddy**


    **Presentation Link**: [Data Speaks E2E](https://docs.google.com/presentation/d/1uTKj7aBFPeK6ticnvwz21gCirP2p7iuUchPJqWu9UME/edit?usp=sharing)

    **Research Paper**: [Data_Speaks, E2E Data to Text with T5, E2E](https://drive.google.com/file/d/1MaVmoKCU_EyB-xjZKvJGwRcSeeYEJgcW/view?usp=sharing)




### Abstract:
This study presents an easy-to-use model tasked for data-to-text generation by extending the T5 [(Raffel et al, 2019)](https://arxiv.org/abs/1910.10683) model architecture with a language modeling head. Model variants allow for trade offs between semantic fidelity, accuracy, and text diversity. The report covers outcomes and analysis with regards to model complexity, resources, and text-generation techniques. The goal is to operate in a true end-to-end fashion with no reliance on intermediate planning steps. With minimal training data, we leverage the Cleaned E2E dataset [(Novikova et al, 2016)](https://arxiv.org/abs/1608.00339v1) to achieve encouraging results across many automated metrics. We hope our findings and design work as a foundation to easily extend this architecture to new datasets and domains.

### Environment Setup:

#### 1. To Create an environment

```
#In project directory
>>> python3 -m venv venv
>>> source venv/bin/activate

#Install all the packages for our project
>>> pip install -r requirements.text

#To close virtual environment
>>> deactivate 
```

#### 2. Convert  `.py` file to `.ipynb` using Jupytext
* Reference: https://jupytext.readthedocs.io/en/latest/using-cli.html

```
jupytext --to notebook notebook.py
```

### Directory Structure

#### Key Folders/Files:


```./src```
Contains all the source code used for various parts of projects. Including utilities, custom classes, EDA, and experiments

```./src/final_model```
Contains 
* `final_model_flow.py`, notebook file to build the final model
* `./model`, contains `.h5` and `config.json` to load the saved model
* `./output`, contains model generated output and metrics.


```./src/metrics_script```
Contains directory provided by shared challenge to compute automated-evaluation metrics.

```./src/experiments/view_results.ipynb```
We have provided a notebook to quickly view results of the generation experiments. You may filter by an experiment type and each individual experiment, or view them all at once across experiment type.


```./data```
We have provided the original E2E and Cleaned E2E data accessible in our repo. For development we have used the HuggingFace provisioning of the data for each of use on demand.

    .
    ├── data
    │   ├── data_sandbox
    │   ├── e2e-cleaning-master
    │   │   ├── cleaned-data
    │   │   ├── partially-cleaned-data
    │   │   └── system-outputs
    │   └── e2e-dataset
    └── src
        ├── classes
        ├── eda
        ├── experiments
        │   ├── base_experiment
        │   ├── base_experiment_t5base
        │   ├── base_experiment_t5large
        │   ├── base_notags
        │   ├── base_random_mrs
        │   ├── batch_size_experiment
        │   ├── beam_temp_exp
        │   ├── final_experiments
        │   ├── gen_experiments
        │   ├── model_training_experiment
        │   ├── no_of_epochs_experiment
        │   ├── sampling_exp
        │   ├── special_tags_exp
        │   ├── t5_base_exp
        │   ├── temp_k_sampling
        │   ├── top_k_sampling
        │   ├── top_p_sampling
        │   └── top_pk_sampling
        ├── final_model
        │   └── output
        ├── metrics_script
        │   ├── example-inputs
        │   ├── metrics
        │   ├── mteval
        │   ├── pycocoevalcap
        │   └── pycocotools
        └── utils


<!-- #endregion -->
