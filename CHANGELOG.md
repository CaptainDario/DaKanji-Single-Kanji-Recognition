# DaKanjiRecognizer - Machine Learning : changelog


## v1.2

new Features:

- recognize:

  - all 漢字 from 漢字検定
  - ひらがな (also historical ones: ゑ, etc.)
  - カタカナ (also historical ones: ヱ, etc.)

Changes:

- handle class imbalance better
- moved fonts into root directory

-------------------------------------------------------------------------

## v 1.1

changes:

- moved from jupyter notebook to jupyter lab
- multi processing for loading the data
- data generator for feeding batches to the CNN
  - multi processed
  - image augmentation

-------------------------------------------------------------------------

## v 1.0

features:

- recognize ~3000 kanji characters
  