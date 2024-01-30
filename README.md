![header](imgs/header.png)

# MassiveFold

## Table of contents
<!-- TOC -->
* [MassiveFold: parallellize protein structure prediction](#massivefold-parallellize-protein-structure-prediction)
* [Installation](#installation)
  * [Steps](#steps)
  * [Jobfile's header building](#jobfiles-header-building)
    * [How to add a parameter](#how-to-add-a-parameter)
* [Usage](#usage)
  * [Inference workflow](#inference-workflow)
  * [Parameters](#parameters)
    * [Parameters in run_massivefold.sh](#parameters-in-run_massivefoldsh)
    * [Parameters in the json file](#parameters-in-the-json-file)
* [massivefold_plots: output representation](#massivefold_plots-output-representation)
  * [Required arguments](#required-arguments)
  * [Facultative arguments](#facultative-arguments)
* [Authors](#authors)
<!-- TOC -->

MassiveFold aims at massively expanding the sampling of structure predictions by improving the computing of AlphaFold 
based predictions. It optimizes the parallelization of the structure inference by splitting the computing on CPU 
for alignments, running automatically batches of structure predictions on GPU, finally gathering all the results in one 
final folder, with a global ranking and various plots.

MassiveFold uses [AFmassive](https://github.com/GBLille/AFmassive), a modified AlphaFold version that integrates diversity 
parameters for massive sampling, as an updated version of Björn Wallner's [AFsample](https://github.com/bjornwallner/alphafoldv2.2.0/).

# MassiveFold: parallellize protein structure prediction
MassiveFold is designed for an optimized use on a GPU cluster because it can automatically split a prediction run into many jobs.  
This automatic splitting is also convenient for runs on a simple GPU server to manage priorities in jobs. All the developments 
were made to be used with a **SLURM** (Simple Linux Utility for Resource Management) workload manager.

![header](imgs/massivefold_diagram.svg)

A run is composed of three steps:  
1. **alignment**: on CPU, sequence alignments is the initiation step (can be skipped if alignments are already computed)

2. **structure prediction**: on GPU, structure predictions follow the massive sampling principle. The total number 
of predictions is divided into smaller batches and each of them is distributed on a single GPU. These jobs wait for the 
alignment job to be over, if the alignments are not provided by the user.

3. **post_treatment**: on CPU, it finishes the job by gathering all batches outputs and produces plots with the 
[MF_plots module](#massivefold_plots-output-representation) to visualize the run's performances. This job is executed only once 
all the structure predictions are over. 

# Installation

MassiveFold was initially developed to run massive sampling with [AFmassive](https://github.com/GBLille/AFmassive) and 
relies on it for its installation.

## Steps

1. **Retrieve MassiveFold**

```bash
# clone MassiveFold's repository
git clone https://github.com/GBLille/MassiveFold.git
```

For AFmassive runs:   
Two additional installation steps are required to use MassiveFold for AFmassive runs:
- Download [sequence databases](https://github.com/GBLille/AFmassive?tab=readme-ov-file#sequence-databases)
- Retrieve the [neural network models parameters](https://github.com/GBLille/AFmassive?tab=readme-ov-file#alphafold-neural-network-model-parameters)

2. **Install MassiveFold**

We use an installation based on conda. The **install.sh** script we provide installs the conda environment using the 
`environment.yml` file. It also creates the file's organization and set paths according to this organization 
in the `params.json` parameters file.

```bash
./install.sh <INSTALLATION_PATH> <data_dir>
```
The <**data_dir**> parameter is the path used in AlphaFold2 installation where the sequence databases are downloaded.

This file tree displays the files' organization after running `./install.sh`. If the <INSTALLATION_PATH> used is 
'..', the tree will be similar to this  :

```txt
.
├── MassiveFold
└── massivefold_runs
    ├── scripts
    │   ├── headers/
    │   ├── batching.py
    │   ├── create_jobfile.py
    │   ├── examine_run.py
    │   ├── get_batch.py
    │   └── organize_outputs.py
    ├──input
    │   └── test_multimer.fasta
    ├── log
    │   └── test_multimer/default
    │       ├── alignment.log
    │       │   ...
    │       └── post_treatment.log
    ├── output
    │   └── test_multimer
    │       ├── default
    │       │   ├── plots/
    │       │   │   ...
    │       │   └── ranking_debug.json
    │       └── msas  
    └── AFmassive_pipeline
        ├── params.json
        ├── run_massivefold.sh
        └── templates/
```

3. **Create header files**  

Refer to [Jobfile's header building](#jobfiles-header-building) for this installation step.

To run MassiveFold in parallel on your cluster/server, it is **required** to build custom jobfile headers for each step. 
They should be named as follows: `{step}.slurm` (alignment.slurm, jobarray.slurm and post_treatment.slurm).  
They have to be added in `<INSTALLATION_PATH>/scripts/headers/` directory or another directory set in the 
`jobfile_headers_dir` parameter of the `params.json` file.  
Headers for Jean Zay cluster are provided as examples to follow (named `example_header_\<step>_jeanzay.slurm`), if you 
want to use them, rename them following the previously mentioned naming convention.  

4. **Set custom parameters**

Each cluster has its own specifications in parameterizing job files. For flexibility needs, you can add your custom 
parameters in your headers, and then in the `params.json` file so that you can dynamically change their values in the json file.  

To illustrate these "special needs", here is an example of parameters that can be used on the french national Jean Zay 
cluster to specify GPU type, time limits or the project on which the hours are used:

Go to `params.json` location:
```bash
cd <INSTALLATION_PATH>/massivefold_runs/scripts/
```
Modify `params.json`:
```json
 "custom_params": {
        "jeanzay_gpu": "v100",
        "jeanzay_project": "<project>",
        "jeanzay_account": "<project>@v100",
        "jeanzay_gpu_with_memory": "v100",
        "jeanzay_alignment_time": "05:00:00",
        "jeanzay_jobarray_time": "15:00:00"
 },
```
And specify them in the jobfile headers (such as here for `jobarray.slurm`) 
```bash
#SBATCH --time=$jeanzay_jobarray_time
#SBATCH -C $jeanzay_gpu_with_memory
```

## Jobfile's header building

The jobfile templates for each step are built by combining the jobfile header that you have to create in 
**MF_scripts/parallelization/headers** with the jobfile body in **MF_scripts/parallelization/templates/**.

Only the headers have to be adapted in function of your computing infrastructure. 
Each of the three headers (`alignment`, `jobarray` and `post treatment`) must be located in the **scripts/headers** 
directory (see [File architecture](#installation) section).

Their names should be identical to:
* **alignment.slurm**
* **jobarray.slurm**
* **post_treatment.slurm**

The templates work with the parameters provided in `params.json` file, given as a parameter to the **run_massivefold.sh** script.  
These parameters are substituted in the template job files thanks to the python library [string.Template](https://docs.python.org/3.8/library/string.html#template-strings).  
Refer to [How to add a parameter](#how-to-add-a-parameter) for parameters substitution.

- **Requirement:** In the jobarray's jobfile header (*massivefold_runs/scripts/headers/jobarray.slurm*) should be stated 
that it is a job array and the number of tasks in it has to be given. The task number argument is substituted with 
the *$substitute_batch_number* parameter.\
For SLURM, the expression should be:
```bash
#SBATCH --array=0-$substitute_batch_number
```
For example, if there are 45 batches, with 1 batch per task of the job array, the substituted expression will be:
```bash
#SBATCH --array=0-44
```
- To store job logs following [Setup](#steps)'s file tree, add these lines in the headers:

In **alignment.slurm**:
```bash
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/alignment.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/alignment.log
```
In **jobarray.slurm**:

```bash
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
```
In **post_treatment.slurm**:
```bash
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/post_treatment.log
#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/post_treatment.log
```
We provide templates for the Jean Zay french CNRS national GPU cluster accessible at the [IDRIS](http://www.idris.fr/), 
that can also be used as examples for your own infrastructure.

### How to add a parameter
- Add **\$new_parameter** or **\$\{new_parameter\}** in the template where you want its value to be set and in the 
"custom_params" section of `params.json` where its value can be specified and changed for each run.

**Example** in the json parameters file for Jean Zay headers:
```json
  "custom_params": {
      "jeanzay_account": "project@v100",
      "jeanzay_gpu_with_memory": "v100-32g",
      "jeanzay_jobarray_time": "10:00:00"
  },
```
Where "project" is your 3 letter project with allocated hours on Jean Zay.

- These parameters will be substituted in the header where the parameter keys are located:

```bash
#SBATCH --account=$jeanzay_account

#SBATCH --error=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log
#SBATCH --output=${logs_dir}/${sequence_name}/${run_name}/jobarray_%a.log

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=1
#SBATCH --array=0-$substitute_batch_number
#SBATCH --time=$jeanzay_jobarray_time
##SBATCH --qos=qos_gpu-dev             # Uncomment for job requiring less than 2 hours
##SBATCH --qos=qos_gpu-t4         # Uncomment for job requiring more than 20h (max 16 GPUs)
#SBATCH -C $jeanzay_gpu_with_memory             # GPU type+memory
```
- Never use single \$ symbol for other uses than parameter/value substitution from the json file.\
To use $ inside the template files (bash variables or other uses), use instead $$ as an escape following 
[string.Template](https://docs.python.org/3.8/library/string.html#template-strings) documentation.

# Usage

Set the [parameters of your run](https://github.com/GBLille/AFmassive?tab=readme-ov-file#running-afmassive) 
in the **AFM_run** section of the `params.json` file for instance:
```json
"AFM_run": {
        "AFM_run_model_preset": "multimer",
        "AFM_run_dropout": "false",
        "AFM_run_dropout_structure_module": "false",
        "AFM_run_dropout_rates_filename": "",
        "AFM_run_templates": "true",
        "AFM_run_min_score": "0",
        "AFM_run_max_batch_score": "1",
        "AFM_run_max_recycles": "20",
        "AFM_run_db_preset": "full_dbs",
        "AFM_run_use_gpu_relax": "true",
        "AFM_run_models_to_relax": "none",
        "AFM_run_early_stop_tolerance": "0.5",
        "AFM_run_bfd_max_hits": "100000",
        "AFM_run_mgnify_max_hits": "501",
        "AFM_run_uniprot_max_hits": "50000",
        "AFM_run_uniref_max_hits": "10000"
    },
```
Activate the conda environment, then launch MassiveFold.
```bash
conda activate massivefold-1.1.0
./run_massivefold.sh -s <SEQUENCE_PATH> -r <RUN_NAME> -p <NUMBER_OF_PREDICTIONS_PER_MODEL> -f <JSON_PARAMETERS_FILE> 
```
Example:
```bash
./run_massivefold.sh -s ../input/test_multimer.fasta -r basic_run -p 67 -f params.json
```
For more help and list of required and facultative parameters, run:
```bash
./run_massivefold.sh -h
```
Here is the help message associated with this command:

```txt
Usage: ./run_massivefold.sh -s str -r str -p int -f str [-m str] [-n str] [-b int | [[-C str | -c] [-w int]] ]
./run_massivefold.sh -h for more details 
  Required arguments:
    -s| --sequence: name of the sequence to infer, same as input file without '.fasta'.
    -r| --run: name chosen for the run to organize in outputs.
    -p| --predictions_per_model: number of predictions computed for each neural network model.
    -f| --parameters: json file's path containing the parameters used for this run.

  Facultative arguments:
    -b| --batch_size: number of predictions per batch, should not be higher than -p (default: 25).
    -m| --msas_precomputed: path to directory that contains computed msas.
    -n| --top_n_models: uses the 5 models with best ranking confidence from this run's path.
    -w| --wall_time: total time available for calibration computations, unit is hours (default: 20).
    -C| --calibration_from: path of a previous run to calibrate the batch size from (see --calibrate).

  Facultative options:
    -c| --calibrate: calibrate --batch_size value. Searches for this sequence previous runs and uses
        the longest prediction time found to compute the maximal number of prediction per batch.
        This maximal number depends on the total time given by --wall_time.
```
## Inference workflow

It launches MassiveFold with the same parameters introduced above but instead of running AFmassive a 
single time, it divides it into multiple batches.

For the following examples, we assume that **--model_preset=multimer** as it is the majority of cases to run MassiveFold 
in parallel.

However, **--model_preset=monomer_ptm** works too and needs to be adapted accordingly, at least the models to use (if not all 
as set by default).

You can decide how the run will be divided by assigning `run_massivefold.sh` parameters *e.g.*:

```bash
./run_massivefold.sh -s ..input/H1144.fasta -r 1005_preds -p 67 -b 25 -f params.json
```

The predictions are computed individually for each neural network model,  **-p** or **--predictions_per_model** allows 
to specify the number of predictions desired for each chosen model.  
These **--predictions_per_model** are then divided into batches with a fixed **-b** or **--batch_size** to optimize the 
run in parallel as each batch can be computed on a different GPU, if available.  
The last batch of each NN model is generally smaller than the others to match the number of predictions fixed by 
**--predictions_per_model**.

***N.B.***: an interest to use `run_massivefold.sh` on a single server with a single GPU is to be able to run massive 
sampling for a structure in low priority, allowing other jobs with higher priority to be run in between.

For example, with **-b 25** and **-p 67** the predictions are divided into the following batches, which is repeated for 
each NN model:

  1.  First batch: **--start_prediction=0** and **--end_prediction=24**
  2.  Second batch: **--start_prediction=25** and **--end_prediction=49**
  3.  Third batch: **--start_prediction=50** and **--end_prediction=67** 

By default (if **--models_to_use** is not assigned), all NN models are used: with **--model_preset=multimer**, 
15 models in total = 5 neural network models $\times$ 3 AlphaFold2 versions; with **--model_preset=monomer_ptm**, 5 
neural network models are used.

The prediction number per model can be adjusted, here with 67 per model and 15 models, it amounts to **1005 predictions 
in total divided into 45 batches**, these batches can therefore be run in parallel on a GPU cluster infrastructure.

## Parameters

### Parameters in run_massivefold.sh

In addition to the parameters displayed with **-h** option, the json parameters file set with **-f** or **--parameters** 
should be organized like the `params.json` file.

### Parameters in the json file

Each section of `params.json` is used for a different purpose.

The **massivefold** section designates the whole run parameters.  

```json
{
"massivefold": 
  {
    "run_massivefold": "",
    "run_massivefold_plots": "",
    "data_dir": "/gpfsdswork/dataset/AlphaFold-2.3.1",
    "jobfile_headers_dir": "./headers",
    "jobfile_templates_dir": "./templates",
    "output_dir": "../output_array",
    "logs_dir": "../log_parallel",
    "input_dir": "../input",
    "predictions_to_relax": "5",
    "models_to_use": ""
  }
}
```
You have to fill the paths in this section. However, the `install.sh` should fill in the majority of them. 
Headers are specified here to setup the run, in order to give the parameters that are required to run the jobs on your 
cluster. Build your own according to the [Jobfile's header building](#jobfiles-header-building) section.

- The **custom_params** section is relative to the personalized parameters that you want to add for your own cluster. 
For instance, for the Jean Zay GPU cluster:
```json
{
  "custom_params": 
    {
      "jeanzay_project": "project",
      "jeanzay_account": "project@v100",
      "jeanzay_gpu_with_memory": "v100-32g",
      "jeanzay_alignment_time": "10:00:00",
      "jeanzay_jobarray_time": "10:00:00"
    }
}
```
As explained in [How to add a parameter](#how-to-add-a-parameter), these variables are substituted by their value when 
the jobfiles are created.

- The **AFM_run** section gathers all the parameters used by MassiveFold for the run (see [AFmassive parameters](https://github.com/GBLille/AFmassive?tab=readme-ov-file#running-afmassive) 
section). All parameters except *--models_to_relax*, *--use_precomputed_msas*, 
*--alignment_only*, *--start_prediction*, *--end_prediction*, *--fasta_path* and *--output_dir* are exposed in this section.  
You can adapt the parameters values in function of your needs.  
The non exposed parameters mentioned before are set internally by the Massivefold's pipeline.

```json
{
  "AFM_run": 
    {
      "AFM_run_model_preset": "multimer",
      "AFM_run_dropout": "false",
      "AFM_run_dropout_structure_module": "false",
      "AFM_run_dropout_rates_filename": "",
      "AFM_run_templates": "true",
      "AFM_run_min_score": "0",
      "AFM_run_max_batch_score": "1",
      "AFM_run_max_recycles": "21",
      "AFM_run_db_preset": "full_dbs",
      "AFM_run_use_gpu_relax": "true",
      "AFM_run_models_to_relax": "none",
      "AFM_run_early_stop_tolerance": "0.5",
      "AFM_run_bfd_max_hits": "100000",
      "AFM_run_mgnify_max_hits": "501",
      "AFM_run_uniprot_max_hits": "50000",
      "AFM_run_uniref_max_hits": "10000"
    }
}
```
Lastly, the **MF_plots** section is used for the MassiveFold plotting module.

```json
  "plots": 
    {
      "MF_plots_top_n_predictions":"5",
      "MF_plots_chosen_plots": "coverage,DM_plddt_PAE,CF_PAEs"
    }
```
# massivefold_plots: output representation

MassiveFold plotting module can be used on a MassiveFold output to evaluate its predictions visually.  

Here is an example of a basic command you can run:
```bash
conda activate massivefold-1.1.0
massivefold_plots.py --input_path=<path_to_MF_output> --chosen_plots=DM_plddt_PAE
```
## Required arguments
- **--input_path**: it designates MassiveFold output dir and the directory to store the plots except if you want them 
- in a separate directory, use *--output_path* for this purpose

- **--chosen_plots**: plots you want to get. You can give a list of plot names separated by a coma 
- (e.g: *--chosen_plots=coverage,DM_plddt_PAE,CF_PAEs*). This is the list of all the available plots:
  * DM_plddt_PAE: Deepmind's plot for predicted lddt per residue and predicted aligned error matrix
  ![header](imgs/plot_illustrations/plddt_PAES.png)
  * CF_plddts: ColabFold's plot for predicted lddt per residue
  ![header](imgs/plot_illustrations/plddts.png)
  * CF_PAEs: ColabFold's plot for predicted aligned error of the n best predictions set with *--top_n_predictions*
  ![header](imgs/plot_illustrations/PAEs.png)
  * coverage: ColabFold's plot for sequence alignment coverage
  ![header](imgs/plot_illustrations/coverage.png)
  * score_distribution:
  * distribution_comparison: ranking confidence distribution comparison between various MassiveFold outputs, typically 
  useful for runs with different parameters on the same sequence.
  ![header](imgs/plot_illustrations/distribution_comparison.png)
  * recycles: ranking confidence during the recycle process (only for multimers)
  ![header](imgs/plot_illustrations/recycles.png)

## Facultative arguments
- *--top_n_predictions*: (default 10), number of best predictions to take into account for plotting
- *--runs_to_compare*: names of the runs you want to compare on their distribution,this argument is coupled with 
**--chosen_plots=distribution_comparison**

# Authors
Nessim Raouraoua (UGSF - UMR 8576, France)  
Claudio Mirabello (NBIS, Sweden)  
Christophe Blanchet (IFB, France)  
Björn Wallner (Linköping University, Sweden)  
Marc F Lensink (UGSF - UMR8576, France)  
Guillaume Brysbaert (UGSF - UMR 8576, France)  

This work was carried out as part of the Work Package 4 of the [MUDIS4LS project](https://www.france-bioinformatique.fr/actualites/mudis4ls-le-projet-despaces-numeriques-mutualises-pour-les-sciences-du-vivant/) 
lead by the French Bioinformatics Institute ([IFB](https://www.france-bioinformatique.fr/)). It was initiated at the 
[IDRIS Open Hackathon](http://www.idris.fr/annonces/idris-gpu-hackathon-2023.html), part of the Open Hackathons program. 
The authors would like to acknowledge OpenACC-Standard.org for their support.
