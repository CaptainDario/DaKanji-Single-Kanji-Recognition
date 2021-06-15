# Single Kanji CNN
This CNN can recognize a single character (Kanji, Hiragana, Katakana, Romaji, Number).
A list of all supported characters can be found [here](./labels.txt).
<br>
## Training
To generate the data necessary to train this CNN, the [single_kanji_data_gen notebook](single_kanji_data_gen.ipynb) is used.
The training can than be done with the [single_kanji_cnn_training notebook](single_kanji_cnn_training.ipynb).<br/>

### fonts 
Because artificial data needs to be generated, as many fonts as possible are required.
Because many fonts do not allow to redistributed them,
for generating data one has to get fonts which include all characters of the JIS Level 2.
50 fonts from [Google fonts](https://fonts.google.com/) and [FreeJapaneseFonts](https://www.freejapanesefont.com/) should be sufficient.
Those have to be stored in [the fonts folder](./fonts).


## Inference
In the releases section pretrained model weights can be found. Also a TensorFlow lite model is available.<br/>

**Input:**
The input should be a *grayscale, 8-bit* image of *any scale*.

**Output:**
A one-hot-vector containing the class probabilities.


## Credits
The data on which the neural network was trained on was kindly provided by [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) <br/>
Papers:<br/>
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* [Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf) <br/>
* [A neural framework for online recognition of handwritten Kanji characters](https://www.researchgate.net/publication/327893142_A_neural_framework_for_online_recognition_of_handwritten_Kanji_characters) <br/>
* [Online Handwritten Kanji Recognition Based on Inter-stroke Grammar](https://www.researchgate.net/publication/4288187_Online_Handwritten_Kanji_Recognition_Based_on_Inter-stroke_Grammar) <br/><br/>