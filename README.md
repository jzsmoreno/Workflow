# Workflow :octocat:
 From Lab to Production Projects using Streamlit

## Setup

To install the dependencies, you can simply follow this steps.

Clone the project repository:
```bash
git clone https://github.com/jzsmoreno/Workflow.git
cd Workflow
```

To create and activate the virtual environment, follow these steps:

**Using `conda`**

```bash
$ conda create -n streamlit python=3.7

# Activate the virtual environment:
$ conda activate streamlit

# To deactivate (when you're done):
(streamlit)$ conda deactivate
```

**Using `virtualenv`**

```bash
# In this case I'm supposing that your latest python3 version is 3.7
$ virtualenv streamlit --python=python3

# Activate the virtual environment:
$ source streamlit/bin/activate

# To deactivate (when you're done):
(streamlit)$ deactivate
```

To install the requirements using `pip`, once the virtual environment is active:
```bash
(streamlit)$ pip install -r requirements.txt
```

#### Running the script

Finally, if you want to run the main script:
```bash
$ streamlit run example/app.py
```
#### Other models

If you want to use another AI model, for example the forecasting model, run the following command line:

```bash
$ streamlit run forecasting/app.py
```
