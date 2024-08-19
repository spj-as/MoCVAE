# MoCVAE

## Setup

1. Clone the repo
    ```bash
    git clone https://github.com/spj-as/MoCVAE.git
    ```
2. Create virtual env
   ```bash
   python3 -m venv venv
   ```
3. Activate virtual env
   ```bash
   source venv/bin/activate
   ```
4. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
5. Generate train and test data
   ```bash
   python main.py preprocess
   ```

## Train

1. Train gagnet model with graph attention and default parameters
    ```bash
    python main.py MoCVAE --name experiment_1 --graph_model gat
    ```

## Dataset Description

> You should run `python main.py preprocess` to generate train and test data. The data will be saved in `data` folder.

