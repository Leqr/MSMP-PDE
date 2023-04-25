# MUTLI-SCALE MESSAGE PASSING NEURAL PDE SOLVERS

Code adapted from <a href="https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers">brandstetter-johannes/MP-Neural-PDE-Solvers</a>, <a href="https://arxiv.org/abs/2202.03376">Link to paper</a>

Implementation for paper <a href="https://arxiv.org/abs/2302.03580">Multi-Scale Message Passing Neural PDE Solvers</a>

### Set up conda environment

source environment.sh

Tasks IDs are E1, E2, E3, WE1, WE2, WE3, KF, KS, kdv, RP, MSWG

### Produce datasets
`python generate/generate_data.py --experiment=TaskID --train_samples=2048 --valid_samples=128 --test_samples=128 --log=True --device=cuda`

###  Train solvers

`python experiments/train.py --device=cuda --experiment=TaskID --model={MP-PDE, BaseCNN, MSMP-PDE, ...} --time_window=25 --log=True`

###  CV solvers replicate i

`python experiments/cv.py --device=cuda --experiment=TaskID --model={MP-PDE, BaseCNN, MSMP-PDE, ...} --time_window=25 --rep=i`

