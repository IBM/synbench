# synbench
Official repo for the paper "SynBench: Task-Agnostic Benchmarking of Pretrained Representations using Synthetic Data"

### Enviroment setup
```
pip install -r requirements.txt
```

### Quickstart 
Evaluate 'vit_base_patch16_224' with SynBench
```
python evaluate_synbench.py --fm_name vit_base_patch16_224 --calculate 1
```
### Cite
```
@article{ko2022synbench,
  title={SynBench: Task-Agnostic Benchmarking of Pretrained Representations using Synthetic Data},
  author={Ko, Ching-Yun and Chen, Pin-Yu and Mohapatra, Jeet and Das, Payel and Daniel, Luca},
  journal={arXiv preprint arXiv:2210.02989},
  year={2022}
}
```
