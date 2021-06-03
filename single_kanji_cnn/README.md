# Single Kanji CNN
This CNN can recognize a single character (Kanji, Hiragana, Katakana, Romaji, Number).
The input should be a grayscale image with width and height of 64.

A list of all supported characters can be found [here](./labels.txt).
<br>
To generate the data necessary to train this CNN, the [single_kanji_data_gen notebook](single_kanji_data_gen.ipynb) is used.
The training can than be done with the [single_kanji_cnn_training notebook](single_kanji_cnn_training.ipynb).<br/>
However in the releases section pretrained models and weights can be found.<br/>


## Credits
The data on which the neural network was trained on was kindly provided by [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) <br/>
Papers:<br/>
* [Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf) <br/>
* [A neural framework for online recognition of handwritten Kanji characters](https://www.researchgate.net/publication/327893142_A_neural_framework_for_online_recognition_of_handwritten_Kanji_characters) <br/>
* [Online Handwritten Kanji Recognition Based on Inter-stroke Grammar](https://www.researchgate.net/publication/4288187_Online_Handwritten_Kanji_Recognition_Based_on_Inter-stroke_Grammar) <br/><br/>