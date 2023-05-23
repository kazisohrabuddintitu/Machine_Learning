# Virtual Environment Setup

* Open a new folder for the project.
* Open the command prompt then run this command:

  `python -m venv venv` 

* Now active the virtual environment:

  `.\venv\Scripts\activate`
  
* Now install the dependencies(download the **[requirements.txt](https://github.com/sharminislamshroddha/machine_learning/blob/main/requirements.txt)** file from my github):

  `python -m pip install -U jupyter matplotlib numpy pandas scipy scikit-learn graphviz`

* Update the dependencies in the requirements file:

  `python -m pip freeze > requirements.txt`

* If you are having trouble activating the venv, set execution polity:
  
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted`

* Install requirements:
  
  `pip install -r requirements.txt`
  
  
* Install jupyter:
  
  `pip install jupyter`
  
* Open Jupyter notebook from command prompt:
  
  `jupyter notebook`
 
