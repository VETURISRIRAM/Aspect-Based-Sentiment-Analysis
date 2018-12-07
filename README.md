# Aspect Based Sentiment Analysis


## Objective 
Given an aspect term (also called opinion target) in a sentence, predict the sentiment label for the aspect term in the sentence.  

## Dataset Information
The training dataset folder contains two .csv training data files. The data-1_train.csv contains header+2313 lines (training examples) and data-2_train.csv contains header+3602 lines. You can open the .csv files in excel to view them. However, sometimes excel may drop few lines from showing (due to the presence of special char symbols in review text) based on its settings. To view full content, you may use any text editor (like Sublime Text, Notepad++ etc.). 

 

### The description of each column in each .csv file is as follows:

Column A: review sentence id (unique id for a training instance).<br />
Column B: review sentence.<br />
Column C: aspect term in the sentence.<br />
Column D: aspect term location  [format: start index -- End index].<br />
Column E: sentiment label.<br />

 
Note:- "," in review sentence (column B) is denoted as "[comma]" to separate it from the column delimiter (",") of .csv file. While parsing (data file reading), you can use "," to split the line into fields (coloums) values.

 

## Implementation
Python implementation of Aspect based Sentiment Analysis. Keep both the training files in the same directory as the python files. Run main.py to see results.

#### The project needs some external library files.<br />
Numpy<br />
MatplotLib<br />
Pandas<br />
NLTK<br />
textblob<br />
SKLearn<br />

These could be installed using packaging tools like PIP. We use VENV environment.
