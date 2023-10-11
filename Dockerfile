FROM  python:3.11

RUN apt-get update && \
    python3 -m pip install --upgrade pip && \
    pip3 install pymoo==0.6.0.1 jax==0.4.18 jaxlib==0.4.18 scipy==1.11.3 matplotlib==3.8.0 pandas==2.1.1 numpy==1.26.0 && \
    pip3 install openai==0.28.1 tensorflow-probability==0.19.0 gpjax==0.7.0 optax==0.1.7 jaxopt==0.8.1

RUN apt-get install -y texlive-latex-base dvipng texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra

RUN mkdir llmbo
WORKDIR /llmbo

# replace with github clone later 
COPY . .


