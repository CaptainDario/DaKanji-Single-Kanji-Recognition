# DaKanjiRecognizer-ML
The machine learning parts of the DaKanjiRecognizer applications.


## Setup development environment
install all dependencies:
```
python -m pip install wheel
```
```
python -m pip install -r requirements.txt
```
Because the database used for this application is needed please follow the setup guide for the [ETL_data_reader package](https://github.com/CaptainDario/ETL_data_reader).

## Kanji recognition CNN
This CNN tries to recognize Kanji characters.
A list of all supported characters can be found [here](./CNN_kanji_only/labels_CNN_kanji_only.txt) and an [desktop application using it here.](https://github.com/CaptainDario/DaKanjiRecognizer-Desktop).
<br>
In the [CNN_kanji](./CNN_kanji) folder you can find the python files necessary to train this model on your own.
When everything is setup correctly `training.py` can be run to train the network. <br>
For a more interactive approach look at the [jupyter notebook](./CNN_kanji/jupyter/DaKanjiRecognizer.ipynb).
In the releases section pretrained weights can be found.

## Credits
### CNN for kanji detection
The data on which the neural network was trained on was kindly provided by [ETL Character Database](http://etlcdb.db.aist.go.jp/obtaining-etl-character-database) <br/><br/>
Papers:<br/>
* [Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks](http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf) <br/>
* [A neural framework for online recognition of handwritten Kanji characters](https://www.researchgate.net/publication/327893142_A_neural_framework_for_online_recognition_of_handwritten_Kanji_characters) <br/>
* [Online Handwritten Kanji Recognition Based on Inter-stroke Grammar](https://www.researchgate.net/publication/4288187_Online_Handwritten_Kanji_Recognition_Based_on_Inter-stroke_Grammar) <br/><br/>

#### other 
[pages: button design](https://codepen.io/kathykato/pen/rZRaNe)
