**Reference Code**

This sub-folder contain several example source code for various domains and application that are useful to boot-strap your development.

The sample code has been either been developed from scratch or has been implemented as exercices from various courses taught in Udemy or Udacity. Majority of the data sets are publicly accessible and were found in Kaggle or links provided in the Udemy/Udacity courses.

For the Udemy courses majority of the code in NLP is from courses taught by the 'thelazyprogrammer' who, in my opinion, goes deep into the concepts and gives a very good coverage of both theoretical and practical aspects. He is my favorite instructor in Udemy. 


**Setting up Python venv



**CUDA Install Notes**
1. Uninstall current torch

pip uninstall torch

2. Install torch with CUDA:

Latest torch version:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

torchtect only with torch 2.2.2 so change according, reduce cuda version also:
pip3 install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121