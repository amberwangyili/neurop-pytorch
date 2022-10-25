# Neural Color Operators for Sequential Image Retouching (Pytorch Implementation) 

*[Yili Wang](https://yili.host), Xin Li, [Kun Xu](https://cg.cs.tsinghua.edu.cn/people/~kun/), Dongliang He, Qi Zhang, Fu Li, Errui Ding*


[[`arXiv`](https://arxiv.org/abs/2207.08080)] [[`project`](https://amberwangyili.github.io/neurop)] [[`doi`](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4401_ECCV_2022_paper.php)]

## Get Started

- Clone this repo

  ```
  git clone https://github.com/amberwangyili/neurop-pytorch
  ```
  
- Download the Dataset from [百度网盘](https://pan.baidu.com/s/1GD1VzZhSoRG6qOQ55u2buQ) (code:jvvq) and unzip in project folder

  ```bash
  tree -L 2 neurop-pytorch/datasets
  # the output should be like the following:
  datasets/
  ├── dataset-dark
  │   ├── testA
  │   ├── testB
  │   ├── trainA
  │   └── trainB
  ├── dataset-init
  │   ├── BC
  │   ├── EX
  │   └── VB
  ├── dataset-lite
  │   ├── testA
  │   ├── testB
  │   ├── trainA
  │   └── trainB
  └── dataset-ppr
      ├── ppr-a
      ├── ppr-b
      ├── ppr-c
      ├── testA
      ├── testM
      ├── trainA
      └── trainM
  ```

- Install Dependencies

  ```bash 
  cd neurop-pytorch/codes
  pip install -r requirements.txt 
  ```

## Test

1. We provide pretrained model weights for MIT-Adobe FiveK and PPR10K in `neurop-pytorch/pretrain_models/`

1. Run command:

   ```bash
   python test.py -config configs/test/<configuaration-name>.yaml 
   ```

2. The evaluation results will be in the `neurop-pytorch/results` folder

## Train

1. Initialization individual neural color operators:
   
   ```bash
   python train.py -config ./configs/init_neurop.yaml 
   ```

2. Finetune with strength predictors:

   ```bash
   python train.py -config ./configs/train/<configuration-name>.yaml 
   ```

