# Emotion analysis and classification of short comments using machine learning techniques
:smiley: :roll_eyes: :pensive: :sob:
This repository contains codes that allow studying supervised machine learning models for short text comment classification tasks.

# Pytho Environment

The project requires Python 3.8 or higher.
You can check the Python version using the command in terminal:
```
python --version
```
If you don't have Python installed on your machine, see: https://www.python.org

# About scripts and files

The repository currently relies on the `main.ipynb`, `oversampling_test.ipynb`, `emotion_analysis.py` scripts:
* `main.ipynb`: This has the main application of the project, it has tasks of pre-processing the texts and classification of them considering the Naive Bayes(NB), Support Vector Machine(SVM) and K-Nearest Neighbors (KNN) models.

* `oversampling_test.ipynb`: This has an isolated case, the application of the main code, performing tests of the Oversampling function that seeks to balance the database, considering that the `dataset.xlsx` file has an amount of unbalanced emotions (classes).

* `emotion_analysis.py`: This one has all the functions necessary for the operation of the other codes, storing functions of pre-processing of the data and also of plots necessary for the evaluation of each model.

* `dataset.xlsx`: This has the data set collected by the authors of the present project, having 173 comments in Portuguese classified as Joy, Sadness and Surprise.

# 
