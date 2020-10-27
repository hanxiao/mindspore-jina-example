Mindspore CPU support

https://github.com/search?q=cpu+repo%3Amindspore-ai%2Fmindspore+extension%3Amd+path%3A%2Fmodel_zoo%2Fofficial&type=Code&ref=advsearch&l=&l=

on MacOS

```
➜  jina-with-mindspore git:(master) ✗ jina hub new
           JINA@37087[I]:
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNNNNNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMWxxxxxxxxxOMMMMMNxxxxxxxxx0MMMMMKddddddxkKWMMMMMMMMMMMMXOxdddxONMMMM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOllllllllllo0MMMMMMMM0olllllllllo0MM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOlllllllllllloWMMMMMdllllllllllllldM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOllllllllllllloMMMM0lllllllllllllllK
MMMMMMMMMMMMKllllllllldMMMMM0lllllllllxMMMMMOllllllllllllllKMMM0lllllllllllllllO
MMMMMMMMMMMMKllllllllldMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMollllllllllllllO
MWOkkkkk0MMMKlllllllllkMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMMxlllllllllllllO
NkkkkkkkkkMMKlllllllloMMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMMMWOdolllllllllO
KkkkkkkkkkNMKllllllldMMMMMMMMWWWWWWWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MOkkkkkkk0MMKllllldXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMWX00KXMMMMXxk0XMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

▶️  /Users/hanxiao/.pyenv/versions/3.7.5/bin/jina hub new
🔧️                            cli = hub
🔧️                            hub = new
🔧️                     output-dir = .
🔧️                      overwrite = False
🔧️                       template = https://github.com/jina-ai/coo
🔧️                           type = pod

You've downloaded /Users/hanxiao/.cookiecutters/cookiecutter-jina-hub before. Is it okay to delete and re-download it? [yes]:
executor_name [The class name of the executor (UpperCamelCase)]: MindsporeLeNet
Select executor_type:
1 - Encoder
2 - Crafter
3 - Indexer
4 - Ranker
Choose from 1, 2, 3, 4 [1]: 1
description [What does this executor do?]: Encoding image into vectors using mindspore
keywords [keywords to describe the executor, separated by commas]: mindspore, lenet
pip_requirements []:
base_image [jinaai/jina]: mindspore/mindspore:1.0.0
author_name [Jina AI Dev-Team (dev-team@jina.ai)]:
author_url [https://jina.ai]:
author_vendor [Jina AI Limited]:
docs_url [https://github.com/jina-ai/jina-hub]:
version [0.0.1]:
license [apache-2.0]:
```

add 
```dockerfile
RUN cd lenet && \
    python train.py --data_path data/fashion/ --ckpt_path ckpt --device_target="CPU" && \
    cd -
```

```
➜  jina-with-mindspore git:(master) ✗ jina hub build MindsporeLeNet/ --pull
           JINA@37470[I]:
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNNNNNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMNNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNNNNNNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNNNWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMWxxxxxxxxxOMMMMMNxxxxxxxxx0MMMMMKddddddxkKWMMMMMMMMMMMMXOxdddxONMMMM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOllllllllllo0MMMMMMMM0olllllllllo0MM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOlllllllllllloWMMMMMdllllllllllllldM
MMMMMMMMMMMMXllllllllldMMMMM0lllllllllxMMMMMOllllllllllllloMMMM0lllllllllllllllK
MMMMMMMMMMMMKllllllllldMMMMM0lllllllllxMMMMMOllllllllllllllKMMM0lllllllllllllllO
MMMMMMMMMMMMKllllllllldMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMollllllllllllllO
MWOkkkkk0MMMKlllllllllkMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMMxlllllllllllllO
NkkkkkkkkkMMKlllllllloMMMMMM0lllllllllxMMMMMOllllllllllllll0MMMMMMWOdolllllllllO
KkkkkkkkkkNMKllllllldMMMMMMMMWWWWWWWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MOkkkkkkk0MMKllllldXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMWX00KXMMMMXxk0XMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM

▶️  /Users/hanxiao/.pyenv/versions/3.7.5/bin/jina hub build MindsporeLeNet/ --pull
🔧️                            cli = hub
🔧️                         daemon = False
🔧️                        dry-run = False
🔧️                      host-info = False
🔧️                            hub = build
🔧️                       password =
🔧️                           path = MindsporeLeNet/
🔧️                   prune-images = False
🔧️                           pull = True
🔧️                           push = False
🔧️                    raise-error = False
🔧️                       registry = https://index.docker.io/v1/
🔧️                      test-uses = False
🔧️                       username =

```

## python -m

echo "export PATH=\"`python -m site --user-base`/bin:$PATH\"" >> ~/.bashrc
pip install --user


```
python -m pytest -s

ENTRYPOINT ["python", "-m", "jina", "pod", "--uses", "config.yml"]
```

## --test-uses


## copy paste requests.on: