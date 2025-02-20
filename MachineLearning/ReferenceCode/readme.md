**Reference Code**

This sub-folder contain several example source code for various domains and application that are useful to boot-strap your development.

The sample code has been either been developed from scratch or has been implemented as exercices from various courses taught in Udemy or Udacity. Majority of the data sets are publicly accessible and were found in Kaggle or links provided in the Udemy/Udacity courses.

For the Udemy courses majority of the code in NLP is from courses taught by the 'thelazyprogrammer' who, in my opinion, goes deep into the concepts and gives a very good coverage of both theoretical and practical aspects. He is my favorite instructor in Udemy. 


**Setting up Python venv

# Steps to create a venv:
1. Run: py -m venv .venv
2. Run: .venv\Scripts\activate
3. Run to verify: where python
4. Run to upgrade pip: py -m pip install --upgrade pip


# Other Commands:
To deactivate venv: deactivate
Install from Requirements: pip install -r .\requirements.txt
To select specific Python Environment:
Goto View -> Command Palette -> Python: Select Interpreter

**CUDA Install Notes**
1. Uninstall current torch

pip uninstall torch

2. Install torch with CUDA:

Latest torch version:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

torchtect only with torch 2.2.2 so change according, reduce cuda version also:
pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121