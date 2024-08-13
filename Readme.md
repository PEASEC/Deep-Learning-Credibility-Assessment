# Deep Learning Models for Real-Time Credibility Assessment in Twitter
A project to provide classifier functionality for activities using neuronal networks.

This project uses the [Git Large File Storage](https://git-lfs.github.com/) (LFS).
Make sure to run the following commands after you cloned the repository:
```
git lfs install
git lfs pull
```

## Contact and Support

If you have any questions, need access to datasets or the complete research data, or if you encounter any bugs, please feel free to contact me!


## Restful Server
In most cases you want a restful server that can be used by external software like the
[Social Media API](https://gitlab.dev.peasec.de/socialmedia/api-core).
This repository contains all necessary data to classify the credibility and hatespeech score using RNNs.
In addition a BERT model to classify the credibility is available.
You can either start the server using Docker or manually using python.

### Using Docker
You do not need any other software except docker. As other services might want to communicate with our server, we should
first set up a docker network:
```bash
docker network create sma-network
```

Then we can start the service using the following command:
```bash
docker-compose up
```

Please note that some configurations can be applied using environmental variables.
They are listed and preconfigured in the `docker-compose.yml` file.

### Using Python
You need Python 3.7 or higher to execute all python scripts.
In addition you need a package manager like pip or conda.

First install all necessary packages:
* Using pip type `pip install -r requirements.txt`.
    You might need to install additional software to use your gpu device for [PyTorch](https://pytorch.org/get-started/locally/).
* Using conda type `conda install -c pytorch -c conda-forge --file requirements-conda.txt`

The requirements file contains ranges of package versions, so each package manager should find
a fitting combination of all packages.


Then you can start the restful API server using:
```bash
python rest.py
```
It will start a server that listens on port 8000 on all interfaces.

There are multiple endpoints available:
* `GET /health`
Check the health of the classifier. This endpoint will always return `{"status": "ok"}`
if the backend is available.

* `GET /classifiers` This endpoint returns the name and the
description of all available classifiers.

* `POST /classify/<classifier>?type=<tweet|activity>`
Classify a list of posts which are provided as Post-Body Format using the
classifier named `<classifier>`. The format of the Post-Body can be chosen
using the optional `?type` parameter. It either accepts a Json representation
of an Activity object or a Json representation of a tweet as returned by the Twitter API.

## Validating Results
You can validate all results stated in the master thesis from Daniel Hartung submitted at 23. Dec. 2020.
There are three available classifiers: _mlp_, _rnn_ and _bert_.
You might need to fetch additional resource data like the training data.
These are stored in a git submodules. To clone it please enter the following command:
```bash
git submodule update --init --recursive
```


### Reproduce training process
You can reproduce each training process by executing the following command:
```
python validate_results.py train <classifier> [feature_type] \
    [--dataset=<default|large|/path/to/dataset>] \
    [--working-dir=<path>]
    [--advanced-timeline] \
    [--ignore-cache]
```

* For `<classifer>` you can use one of the following values:
    
    | Value | Description                                             |
    | ----- | ------------------------------------------------------- |
    | mlp   | Use a default multi layer perceptron for classification |
    | rnn   | Use a recurrent neural network for classification       |
    | bert  | Use the bert language model for classification          |


* For `<feature_type>` the following basis features are available

    | Value     | Description                                             |
    | -----     | ------------------------------------------------------- |
    | (empty)   | Use basis classifier. Not available for mlp.                                  |
    | user      | Only use features associated with the user (e.g. status count)                |
    | tweet     | Only use features associated with the tweet (e.g. creation time)              |
    | text      | Only use manuel selected features from the tweet text (e.g. number of letters)|
    | timeline  | Use previous posts of a user to generate additional features.                 |
    
    You can also combine features using dashes (`-`).
    The feature `tweet-text-user` will use tweet, text and user features.
    
* The `--dataset` configuration allows you to select a specific dataset.
    You can either provide _default_, _large_ or a path to a custom dataset.
    The default value is _large_.

* The `--working-dir` configuration allows to define where to store the results
    from the training process. By default it will be saved to `./build/<dataset-name>`
    where dataset name represents the name or path of the chosen directory.
    
* The `--advanced-timeline` switch only works if the timeline is used as one feature.
    If the switch is present, the classifier will switch to a different method to extract timeline features.
    
* The `--ignore-chace` switch allows you to ignore cached results and force to recreate them.

For more options see [Additional parameter](#additional-parameter).

### Reproduce external validation process

An already trained classifier can be validated on a custom dataset using the following command:

 ```
python validate_results.py train <classifier> <feature_type> \
    --dataset=<path> \
    [--working-dir=<path>] \
    [--advanced-timeline] 
    [--train-set | --dev-set | --test-set]
```

The settings for `classifier`, `feature_type`, `--dataset` and `--advanced-timeline` are the same as
mentioned above.

The parameter `--working-dir` must be the same as provided in the train process.
If the validation set has a splitting information for train. test and dev sets, the optional
arguments `--train-set`, `--dev-set` or `--test-set` can be provided to test only a subset.

For more options see [Additional parameter](#additional-parameter).


### Measure time
The duration of classification tasks of an already trained classifier can be validated using the following command:

```
python validate_results.py measure <classifier> <feature_type> \
    --dataset=<path> \
    [--working-dir=<path>] \
    [--advanced-timeline]
    [--train-set | --dev-set | --test-set]
    [--wait] 
```

If `--wait` is provided the program waits for user input before it ends.
This can be used to analyse ram usage.


### Analyse Dataset
A dataset can be analysed using the following command:
```
python validate_results.py analyze --dataset=<path>
```


### Additional parameter
* General:
    * `--sample=<count>` If provided a random subset with `count` elements will be extracted from the dataset and used

* For `RNN`:
    * `--memory` If set the classifier will read the embedding file once and keep it into the memory
        (may causes high RAM uses). Otherwise the classifier index the file and will read most words from the file.
    
    * `--embeding=</path/to/embedding>` (Default: `resources/embeddings/glove.twitter.27B.50d.enriched.txt`) The
        path of the embedding file
        
* For `BERT`:
    * `--bert-model=<path>` The path to the fine tuned bert model


### Reproducing finetuning bert model

To reproduce the finetuning process of the bert model you need to use a Jupyter-notebook. 
[Google Colab](https://colab.research.google.com) provides free processing resources and was used in this project.
You can upload `./FineTuneBert.ipynb` and execute the file.
All necessary information are provided inside the file.

## Authors 

Daniel Hartung, Marc-André Kaufhold, Markus Bayer, and Christian Reuter
