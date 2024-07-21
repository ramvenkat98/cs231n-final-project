This repo contains the code for our CS231N project submission where we generate a synthetic dataset for figure-based Math problems and fine-tune a vision language model (VLM) on it. We evaluate on the MathVista dataset and observe performance improvements.

There are two parts of this repo that do not contain original code:
* The finetune_changes_copy/ directory: this contains files from the InternLM-XComposer repo (https://github.com/InternLM/InternLM-XComposer/tree/main) that we slightly modify in order to meet our fine-tuning needs. These are mainly just changes to the finetuning configs (we add two sample configs that we use for LoRA and DoRA), and a minor change to the finetuning script to support DoRA (in addition to LoRA).
* The utils.py file - this contains some helper functions, some of which are taken from the InternLM repo mentioned above, and some which are taken from the MathVista repo: https://github.com/lupantech/MathVista.

## Abstract
In this work, we present GAMMAS: a pipeline that uses GPT-4 and code-generation to synthesize a fine-tuning and evaluation dataset containing 860 bar and line charts, and relevant mathematical questions. We then use this dataset to fine- tune the InternLM-XComposer2-VL-1.8B model. We show a 6.5% and 12.5% increase in performance on the bar and line chart tasks in the MathVista benchmark without a loss in overall performance. We also note that performance on other categories of tasks in MathVista increases as well.

## Poster
<img width="1202" alt="CS231N_Poster" src="https://github.com/ramvenkat98/cs231n-final-project/blob/main/CS231N_Poster.pdf">


## Report
Our report can be found [here](https://github.com/ramvenkat98/cs231n-final-project/blob/main/CS231_Final_Report.pdf).
