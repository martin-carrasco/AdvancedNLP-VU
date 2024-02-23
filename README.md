# AdvancedNLP-VU
Repository for the course of Advanced NLP at VU Amsterdam

# General Information
The repository is composed of 3 main folder, each of them representing an individual assignment. There will be
no calls between files on the assignments, they are all standalone.

# Instructions

## Assignment 1
## Assignment 2
### Folder structure
The `data` folder organizes the files that will be handled for the SRL task. It allows for models to be saved and 
intermediate data to be saved as well during execution. Raw data that wants to be fed should be put in the `raw` subfolder from which it can be accessed when calling `main.py`, it should follow the `.conllu` format specification. The `preproccesed` subfolder is an intermediary folder where results are stored however you do not need to worry about it. If you want to inspect the data extracted from the `.conllu` file, you can look there. The `input` folder will contain the output of preprocessing the data and can also be directly used to put processed data from other means. When using the `-i` flag on `main.py` it will look there. The `models` folder contains the pickled binary data of the models and the `preds` folder contains the CSV that are output by a model with the true label next to the predicted label

### Usage
The main file is `main.py`, it can be called from the command line with the arguments needed. Since it does everything from parsing to evaluating results, there are some instructions needed. Arguments will be outline below to know how to call the program. **NOTE** Preprocessing takes a while, so be ready to wait about 15 minutes.

+ `op_mode`: The first argument is **mandatory**, it specifies if the program will run on *preproc* or *model* mode.
    + Ex. ```python main.py preproc ... ```
+ `-i`: The second argument is denoted with `-i`, it specified the input filename. Depending on your operation it will look on the corresponding subfolder. If you are on mode `preproc` it will look for it in raw, if you are training or predicting with a model it will look in `input`.
    + Ex. ```python main.py preproc -i raw_1.conllu```
    + Ex. ```python main.py model -i data.tsv ...```
+ `-m`: This optional argument only applies with `op_mode` is on `model`. It specifies the mode of operation of the model, either `train` or `predict`.
    + Ex. ```python main.py model -i data.tsv -m train ...```
+ `-t`: This optional argument only applies with `op_mode` is on `model`. It specifies what the task to solve is going to be, either `identify` or `classify`.
    + Ex. ```python main.py model -i data.tsv -m train -t identify```
+ `-mf`: This optional argument only applies with `op_mode` is on `model` and on the `predict` model operation. It specifies the name of the model as it appears in the `models` directory.
    + Ex. ```python main.py model -i data.tsv -m predict -t identify -mf model_ident.pkl```