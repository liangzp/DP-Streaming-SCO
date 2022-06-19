# On Optimal Private Online Stochastic Optimization and High Dimensional Decision Making

This repository is the official implementation of [On Optimal Private Online Stochastic Optimization and High
Dimensional Decision Making]() by Yuxuan Han, Zhicong Liang, Zhipeng Liang, Yang Wang, Yuan Yao and Jiheng Zhang.

## Requirements:

Requires python 3, numpy, matplotlib, etc.
Please use the following command to install the dependencies:
```setup
pip install -r requirements.txt
```

## Citation:
If you wish to use our repository in your work, please cite our paper:

BibTex:
```
@inproceedings{han2022dpsteaming,
  title={Optimal Private Streaming SCO in $\ell_p$-geometry with Applications in High Dimensional Online Decision Making},
  author={Han, Yuxuan and Liang, Zhicong and Liang, Zhipeng and Wang, Yang and Yao, Yuan and Zhang, Jiheng},
  booktitle={International Conference on Machine Learning},
  year={2022},
  organization={PMLR}
}
```

Any question about the scripts can be directed to the authors <a href = "mailto: zliangao@connect.ust.hk"> via email</a>.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Generating the figures in the paper:

For generating the figures in the paper please execute the following codes:
1. run the instances to generate the required experiment results data
```
nohup bash execute.sh &
```

2. run **summarize.py** to collect all the statistic from the experiments meta-data
```
python3 summarize.py
```
3. run **plot-curves-paper-p1.5.ipynb/plot-curves-paper-pinf.ipynb/plot-curves-bandit.ipynb** to generate the figures and tables for the "p=1.5"/"p=inf"/"bandit" part.

## Generating the table in the paper:
 
run **summarize_table.ipynb** notebook to generate the Table 1 and 2 in the paper.