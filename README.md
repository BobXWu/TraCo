## **Check our latest topic modeling toolkit [TopMost](https://github.com/bobxwu/topmost) !**

# Code for On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling (AAAI 2024)


[AAAI 2024](https://arxiv.org/pdf/2401.14113.pdf)

## Usage

### 1. Prepare environment

    python==3.8.0
    pytorch==1.7.1
    gensim==4.3.0
    scipy==1.5.2
    scikit-learn==0.24.2
    tqdm
    pyyaml


### 2. Train and evaluate the model

We provide a shell script under `./TraCo/scripts/run.sh` to train and evaluate our model.

Change to directory `./TraCo`, and run command as


    ./scripts/run.sh TraCo NYT 10-50-200


Other datasets are available in [TopMost](https://github.com/BobXWu/TopMost/tree/main/data).


## Citation

If you want to use our code, please cite as

    @inproceedings{wu2024traco,
        title        = {On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling},
        author       = {Wu, Xiaobao and Pan, Fengjun and Nguyen, Thong and Feng, Yichao and Liu, Chaoqun and Nguyen, Cong-Duy and Luu, Anh Tuan},
        year         = 2024,
        booktitle    = {Proceedings of the AAAI Conference on Artificial Intelligence},
        url          = {https://arxiv.org/pdf/2401.14113.pdf}
    }
