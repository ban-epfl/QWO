# QWO: Speeding Up Permutation-Based Causal Discovery in LiGAMs

In this project, you can find the official codes of the NeurIPS 2024 paper ["QWO: Speeding Up Permutation-Based Causal Discovery in LiGAMs"](https://openreview.net/forum?id=BptJGaPn9C&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2024%2FConference%2FAuthors%23your-submissions)) and instructions on how to run them. The codes are in python. 

Also you can play with code for your experiments via this [colab notebook](https://colab.research.google.com/drive/1fJv34nxoCOXKDJj6xuTfvrekQ-0AhuqA?usp=sharing).

## Requirements

To install requirements you need to run the following command:

```setup
pip install -r requirements.txt
```


## Run an experiment

To run a new experiment you only need to change the *config.py* file and then run the following command:

```setup
python3 main.py
```

To modify the data generation process, you can specify the number of variables, the number of data points, the average degree of each node in the Erdos-Renyi graph, and the type of noise in the model (options include Gaussian, exponential, and Gumbel) in the *config.py* file. Additionally, to run an experiment on your existing data, simply insert the path to your pickle file containing the data matrix in CONFIG.path_to_data.

The search method and its parameters can be specified using CONFIG.search_method and CONFIG.search_params.

## Results  

The following figure from the paper compares the accuracy and execution time between our method and the state-of-the-art BIC-based score, evaluated on both search methods and varying numbers of variables. $ERn$ denotes the random graph constructed by the Erdős–Rényi method with an average node degree of $n$. For the definition of metrics, refer to the experiments section of the paper. As illustrated in the figure, across all settings, QWO’s accuracy is higher with the Hill Climbing search method and relatively good with GRaSP, while demonstrating a significant speed improvement over BIC in both search methods.

![](figs/ER2.png)
![](figs/ER3.png)
![](figs/ER4.png)

## Citation 
To cite our paper, please use the following bibtex entry:

```
@article{shahverdikondori2024qwo,
  title={QWO: Speeding Up Permutation-Based Causal Discovery in LiGAMs},
  author={Shahverdikondori, Mohammad and Mokhtarian, Ehsan and Kiyavash, Negar},
  journal={arXiv preprint arXiv:2410.23155},
  year={2024}
}
```
