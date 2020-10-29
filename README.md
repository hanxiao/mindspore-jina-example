# Mindspore-Powered Neural Search in Jina

This is the code repo for the [blog post "Mindspore-Powered Neural Search in Jina"](https://hanxiao.io/2020/10/28/Mindspore-powered-Neural-Search-in-Jina/).

## Preliminaries

- Mac OS or Linux
- Python 3.7, 3.8
- [Jina 0.7+ with Hub extenstion (i.e. `pip install "jina[hub]"`)](https://get.jina.ai)
- Docker

## Build the Jina Hub Image

```bash
git clone https://github.com/hanxiao/mindspore-jina-example.git

jina hub build MindsporeLeNet/ --pull --test-uses
```

## Use in Jina Hello-World

```
jina hello-world --uses-index helloworld.flow.index.yml --uses-query helloworld.flow.query.yml
```