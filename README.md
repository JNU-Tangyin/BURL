# BURL

Behavior-aware URL embedding

In the context of the Internet, the representation of URLs have become the core for almost all downstream analysis, including detecting malicious websites, classifying websites, analyzing customer behavior, and content recommendation. To achieve these goals, various techniques have been developed, typically using word embedding techniques to represent URL components as vectors. These methods have enjoyed the success of accuracy, efficiency, and scalability. However, neglecting the behavioral patterns among consecutive requests in a session, these methods still face challenges such as interpretability and loss of semantics when dealing with various data-sets. In this study, we propose a general purposed Behavior-aware URL (BURL) embedding approach, which combines contrastive learning with URL embedding and demonstrates the superiority of the proposed method through exploration of various downstream tasks. Additionally, a downstream task is introduced to improve the precision of the recommendation system by accurately identifying active behaviors of apps on a private data sets.



## Files & Folders

- **datasets**   the datasets to train and test. Please note that due to limited space, the real large datasets should be downloaded from certain websites, as the `datasets.ipynb` instructs the way of processing raw datasets. 


- **methods** includes baseline algorithms and ours. Please note that `burl` is our algorithm. DRLB, Uniform, Normal, Lin, and Gamma are included in the compare experiment. AC_GAIL_agent, PPO_GAIL_agent, and PPO_agent are included in the ablation experiment.

- **README.md** this file

- **ablation_exp.py**    to do ablation study

- **compared_exp.py**    to do comparison among different baselines (DRLB, Uniform, Normal, Lin, and Gamma)

- **globals.py**  global variables

- **main.py**    main entrance of the experiments. to envoke `run.py`, `post.py`, and `plot.py`

- **plot.py**  read `final.csv`, plot the figures as .pdf used for the paper to store in `figures` folder, and generate  latex tables as .tex files for the papers.

- **post.py**  post-process,  to put together all the intermediate results in to one `final.csv` file.

- **preprocess.py**   read data from `datasets` folder, and preprocess for compare experiment and ablation experiment.

- **requirements.txt**  for install the conda virtual env.


## Usage

1. Install Python 3.9. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

Another more elegant way to reproduce the result, of course, is use conda virtual environment, as widely appreciated. Typically by entering the following command before the above pip installation:

```shell
conda create -n BURL python=3.9
```

We are not going to discuss the details here.

2. Prepare Data. 

Download the original datasets from [IPINYOU](https://contest.ipinyou.com/) and [YOYI](https://apex.sjtu.edu.cn/datasets/7). Process the raw dataset according to the instruct in `datasets.ipynb` in the `datasets` folder. Considering the space limitations, we provide sampled datasets for users to directly invoke and experiment with, which also produce the same results.

3. Train and evaluate model. You can adjust parameters in global.py and reproduce the experiment results as the following examples:

```python
python3 main.py
```

As a scheduler, `main.py` will envoke `run.py`, `post.py`, and `plot.py` one by one, the functions of which are introduced in the **Files & Folders** section. 

4. Check the results
- results are in .csv format at `results` folder, which are later combined together to a `final.csv` for plotting purpose.
- figures are at `figures` folder
- latex tables are at `tables` folder

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{li2024burl,
  title={Behavior-aware URL embedding representation learning},
  author={Zeyan Li, Shengda Zhuo, Yin Tang},
  year={2024},
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Zeyan Li <lzy12345678900823@163.com>
- Yin Tang <ytang@jnu.edu.cn>

Or describe it in Issues.

## Acknowledgement

This work is supported by National Natural Science Foundation of China (62272198) and by Guangdong Provincial Science and Technology Plan Project (No.2021B1111600001).
