<!-- #region -->
# EDA

`t5_exploration.py` - Goal of this notebook is to use the T5 tokenizer to parse the training and validation data. Consider the spread of token lengths and understand what are initial numbers we can experiment with in terms of pre-processing our data as inputs for **T5**. 
Length's are important because they directly affect model performance, generation, and compute time.

`train_sentiment_analysis.py` - Goal of this notebook is to use the pre-built **HuggingFace Distill-BERT** sentiment analysis pipeline on our training and validation data. We did this to understand if there was any implicit bias in our data that would affect our model to generate text that would be predominantly positive or negative.

`cleaned_e2e_eda.py ` - Goal of this notebook is to do an initial analysis of the data we are dealing with, get a sense of the tags involved, the length of entries and gauge the overall spread.


