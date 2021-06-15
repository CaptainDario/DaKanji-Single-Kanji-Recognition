# DaKanji-ML
This repository contains all machine learning models used in DaKanji. <br/>
For every model there is a separate folder.

|      folder name |                     description                     |
| :--------------: | :-------------------------------------------------: |
| single_kanji_cnn | CNN which can detect single hand written characters |

If you are interested in using one of the models go to their folder.
Every folder has more details on the model itself.

## Setup development environment
install all dependencies:
```
python -m pip install wheel
```
```
python -m pip install -r requirements.txt
```

### fonts 
Because artificial data needs to be generated, as many fonts as possible are required.
Because many fonts do not allow to redistributed them,
for generating data one has to get fonts which include all characters of the JIS Level 2.
50 fonts from [Google fonts](https://fonts.google.com/) and [FreeJapaneseFonts](https://www.freejapanesefont.com/) should be sufficient.
Those have to be stored in [the fonts folder](./fonts).

