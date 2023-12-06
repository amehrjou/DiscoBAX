# DiscoBAX - Discovery of optimal intervention sets in genomic experiment design

This package named "discobax" contains the codebase for the paper "DiscoBAX - Discovery of optimal intervention sets in genomic experiment design". In the following we provide a short instruction on how to run the code to reproduce the results of the paper.

## Abstract 

The discovery of therapeutics to treat genetically-driven pathologies relies on identifying genes involved in the underlying disease mechanism. With billions of potential hypotheses to test, an exhaustive exploration of the entire space of potential interventions is impossible in practice. Sample-efficient methods based on active learning or Bayesian optimization bear the promise of identifying targets of interest using as few experiments as possible. However, genomic perturbation experiments typically rely on proxy outcomes measured in biological model systems that may not completely correlate with the results of interventions in humans. In practical experiment design, one aims to find a set of interventions that maximally move a target phenotype via a diverse mechanism set to reduce the risk of failure in future stages of trials. To that end, we introduce  — a sample-efficient algorithm for genomic intervention discovery that maximizes the desired movement of a phenotype while covering a diverse set of underlying mechanisms. We provide theoretical guarantees on the optimality of the approach under standard assumptions, conduct extensive experiments in synthetic and real-world settings relevant to genomic discovery, and demonstrate that DiscoBax outperforms state-of-the-art active learning and Bayesian optimization methods in this task. Better methods for selecting effective and diverse perturbations in biological systems could enable researchers to discover novel therapeutics for many genetically-driven diseases.

## Install

```bash
pip install -r requirements.txt
pip install discobax
```

## Use

### How to Run the experiment with toy (synthetic data)

In the experiments on the synthetic datasets, we aim to illustrate the benefits of our suggested method (DiscoBAX) relative to other baselines in a setting where the ground truth target function is known. 

Running experiments on the synthetic datasets can be achieved as follows:

```bash
python discobax/apps/toy_experiment.py \
            --cache_directory=$cache_directory \
            --output_directory=$output_directory \
            --save_fig_dir=$saving_dir \
            --model_name="gp" \
            --acquisition_function_name=$acquisition_function \
            --acquisition_batch_size=15 \
            --num_active_learning_cycles=30 \
            --dataset_name=$dataset_name
```
where the above variables are defined as follows:
- $cache_directory is the location on disk where the GeneDisco datasets will be stored
- $output_directory is the location on disk where the detailed experimental results across cycles will be stored
- $saving_dir is the location in which output figures will be stored
- $acquisition_function is the name of the acquisition function of interest (eg., "random", "topk_bax", "levelset_bax", "subsetmax_bax_additive")
- $dataset_name is the name of the synthetic dataset (eg., "mog", "sine", "sine2d")


### How to Run the experiment with real-world genetic data on the GeneDisco benchmark

In the experiments based on the GeneDisco benchmark [1], we aim to identify a diverse set of genomic interventions (eg., CRISPR gene knock-off) with high impact on the phenotype of interest. The benchmark is comprised of 5 different assays ($dataset_name listed below) and provides also a dataset for the initial intervention represention (eg., Achilles). In each experiments, we performed 25 consecutive batch acquisition cycles (with batch size 32). All experiments are repeated 10 times with different random seeds (eg., 1000, 2000, 3000).

Running experiments on the GeneDisco benchmark can be achieved as follows:

```bash
python discobax/apps/genedisco_experiment.py \
            --cache_directory=$cache_directory \
            --output_directory=$output_directory \
            --performance_file_location=$performance_file_location \
            --model_name="bayesian_mlp" \
            --acquisition_function_name=$acquisition_function \
            --acquisition_batch_size=32 \
            --num_active_learning_cycles=25 \
            --feature_set_name="achilles" \
            --dataset_name=$dataset_name \
            --seed=$seed_name
```
where the above variables are defined as follows:
- $cache_directory is the location on disk where the GeneDisco datasets will be stored
- $output_directory is the location on disk where the detailed experimental results across cycles will be stored
- $performance_file_location is the path to the file where aggregated performance metrics are saved
- $acquisition_function is the name of the acquisition function of interest (eg., "random", "topk_bax", "levelset_bax", "subsetmax_bax_additive")
- $dataset_name is the name of the GeneDisco assay, to be chosen within the following list: "schmidt_2021_ifng", "schmidt_2021_il2", "zhuang_2019_nk", "sanchez_2021_tau" or "zhu_2021_sarscov2"
- $seed_name is the random seed for pseudo-random number generation (eg., 1000, 2000, 3000)

## Reproducibility
For exact reproduciblility of GeneDisco results, please download the datasets and clusters from the following zenodo [location](https://zenodo.org/records/10202590), and store them in the same `cache_directory` location as discussed above. All experiments were conducted on A100 GPUs.

## References
```
Mehrjou, A., Soleymani, A., Jesson, A., Notin, P., Gal, Y., Bauer, S., & Schwab, P. (2022). GeneDisco: A Benchmark for Experimental Design in Drug Discovery. ICLR 2022.
```
