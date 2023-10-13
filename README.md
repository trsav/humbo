

<!-- Do the above, but make the image on the right of the readme -->

<p align="center">
  <img src="llmbo_dalle.png" alt="LLM-BO" width="300"/>
</p>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Contains the code for the workshop paper: _Expert-guided Bayesian Optimisation for Human-in-the-loop Experimental Design of Known Systems_

<!-- TODO ADD REMOTE OR LOCAL LLM EXECUTION -->
# Instructions

0. Request and download Llama 2  (13B or 70B) (https://ai.meta.com/llama/)

1. Clone Git repository & install submodules
```
$ git clone --recurse-submodules git@github.com:trsav/llmbo.git
```
2. Create conda environment
```
$ conda env create -f environment.yml
```


<!-- 

2. Build Docker container from ```Dockerfile``` (~5 mins)
```
$ sudo docker build --tag llmbo .
```
3. Run container with volume to store data
```
$ sudo docker run -i -t --mount source=llmbo_volume,target=/llmbo llmbo
``` -->
