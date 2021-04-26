# SIGIR'21 - Adapting Interactional Observation Embedding for Counterfactual Learning to Rank

This project implements Interactional Observation-based Model (IOBM), based on [ULTRA toolbox](https://github.com/ULTR-Community/ULTRA/).

## Usage

Here we give the simple instructions for our project. Please see [ULTRA](https://github.com/ULTR-Community/ULTRA/) for more details about this framework.

**Install Datasets:**
```bash
cd example/Yahoo # or example/MSLR_WEB30K
bash offline_exp_pipeline.sh
```

**Select Important Features:**
```bash
python feature_importance.py ./Yahoo_letor/tmp_data/
```

**Run Experiments:**
```bash
python run.py # Please see the arguments in this file for the whole experiment settings.
```

## Citation

```
@inproceedings{iobm,
  author = {Mouxiang Chen, Chenghao Liu, Jianling Sun and Steven C.H. Hoi},
  title = {Adapting Interactional Observation Embedding for Counterfactual Learning to Rank},
  year = {2021},
  booktitle = {Proceedings of the 44th International {ACM} {SIGIR} Conference on Research and Development in Information Retrieval {(SIGIR-21)}}
}
```
