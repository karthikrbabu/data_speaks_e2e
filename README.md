<!-- #region -->
# data_speaks_e2e
Using the E2E dataset, and T5 language model, we take on the data-to-text problem of automatically producing text from non-linguistic input such as database records.



### Environment Setup:


#### 1. To Create an environment

```
#In project directory
>>> python3 -m venv venv
>>> source venv/bin/activate

#Install all the packages for our project
>>> pip install -r requirements.text

#To close virtual environment
>>> deactivate 
```

#### 2. Convert  `.py` file to `.ipynb` using Jupytext
* Reference: https://jupytext.readthedocs.io/en/latest/using-cli.html

```
jupytext --to notebook notebook.py
```


<!-- #endregion -->
