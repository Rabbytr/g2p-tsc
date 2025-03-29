## Generalized Phase Pressure (G2P) Traffic Signal Control

![Document](https://img.shields.io/badge/docs-in_progress-violet)
![Implementation](https://img.shields.io/badge/implementation-python-blue)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2503.20205-red)](https://arxiv.org/abs/2503.20205)
[![Py_version](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/)
![License](https://img.shields.io/badge/License-None-lightgrey)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CityFlow](https://img.shields.io/badge/CityFlow-05bca9?style=for-the-badge)](https://cityflow-project.github.io/)
![TSC](https://img.shields.io/badge/TSC-EA4C89?style=for-the-badge)

### Usage

Run G2P control:
```shell
python run_test.py
```

Run G2P-MPLight:
```shell 
python run.py --agent g2p_mplight
```

Run G2P-CoLight:
```shell 
python run.py --agent g2p_colight
```

### Citations
```text
@misc{liao2025g2p,
      title={Generalized Phase Pressure Control Enhanced Reinforcement Learning for Traffic Signal Control}, 
      author={Xiao-Cheng Liao and Yi Mei and Mengjie Zhang and Xiang-Ling Chen},
      year={2025},
      eprint={2503.20205},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20205}, 
}
```
