<!-- include llmbo_dalle.png as an image centered  -->

<p align="center">
  <img src="llmbo_dalle.png" alt="LLM-BO" width="300"/>
</p>


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Instructions

1. Clone Git repository & install submodules
```
$ git clone --recurse-submodules git@github.com:trsav/llmbo.git
```

2. Build Docker container from ```Dockerfile``` (~5 mins)
```
$ sudo docker build --tag llmbo .
```
3. Run container with volume to store data
```
$ sudo docker run -i -t --mount source=llmbo_volume,target=/llmbo llmbo
```
