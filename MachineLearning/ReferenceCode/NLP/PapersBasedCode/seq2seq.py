# This code has several specific dependencies so it is better
# to execute in a venv
# Steps to create a venv:
# 1. Run: py -m venv .venv
# 2. Run: .venv\Scripts\activate
# 3. Run to verify: where python
# 4. Run to upgrade pip: py -m pip install --upgrade pip
# Other Commands:
# To deactivate venv: deactivate

# torch and torchtext has compatibility issues.
# Only install using this: pip install torch==2.2.2 torchtext==0.17.2
# or use: pip install -r .\requirements.txt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm
import evaluate

# The .py file is mainly for testing as it is easier to find 
# and fix errors here than in the .ipynb


