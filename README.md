# DaKanji Single Kanji Recognition

This CNN can recognize a single character (Kanji, Hiragana, Katakana).
A list of all supported characters can be found [here](./labels.txt).

## Apps which use this model:
| name | android | iOS |
|:---:|:---:|:---:|
| DaKanji| x | x |

## Training

To generate the data necessary to train this CNN, the [single_kanji_data_gen notebook](single_kanji_data_gen.ipynb) is used.
The training can than be done with the [single_kanji_cnn_training notebook](single_kanji_cnn_training.ipynb).

## Inference

In the releases section pretrained model weights can be found. Also a TensorFlow lite model is available.

**Input:**
The input should be a *grayscale* image of *any size*.

**Output:**
A one-hot-vector containing the class probabilities (lines up with `labels.txt`).

## Setup development environment

install all dependencies:

``` python
python -m pip install wheel
python -m pip install -r requirements.txt
```

Now you should follow model specific setup steps.
For this look at the README for the model you are interested in.

## Usage in your software

I put lots of effort and time into developing this model and hope that it can be used in many apps.
If you decide to use this machine learning model please give me credit like:
`Character recognition powered by machine learning from CaptainDario (DaAppLab)`
It would also be nice if you open an issue and tell me that you are using this model.
Than I would add your software to the [test](#App-which-use-this-model)

## Credits

The data on which the neural network was trained on was kindly provided by [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database)

[The KanjiVG dataset](https://kanjivg.tagaini.net/) was used to generate "handwritten" kanjis
Papers:

* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* [Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf)
* [A neural framework for online recognition of handwritten Kanji characters](https://www.researchgate.net/publication/327893142_A_neural_framework_for_online_recognition_of_handwritten_Kanji_characters)
* [Online Handwritten Kanji Recognition Based on Inter-stroke Grammar](https://www.researchgate.net/publication/4288187_Online_Handwritten_Kanji_Recognition_Based_on_Inter-stroke_Grammar)
