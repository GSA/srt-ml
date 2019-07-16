# Tokenization

For the SageMaker pilot, we’re going to use the [BlazingText algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html) to perform binary classification on text documents in the *train with file mode*, so our training data needs to be in the [RecordIO](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html#bt-inputoutput) format:

```

__label__4  linux ready for prime time , intel says , despite all the linux hype , the open-source movement has yet to make a huge splash in the desktop market . that may be about to change , thanks to chipmaking giant intel corp .

__label__2  bowled by the slower one again , kolkata , november 14 the past caught up with sourav ganguly as the indian skippers return to international cricket was short lived . 
```

`Tokenization.ipynb` converts [pre-labeled documents](https://drive.google.com/drive/folders/1jSlRzeZuKj4RRUrgrjXoVcQXsrMtZfB4) into this RecordIOformat before pushing them to an S3 bucket.

## Getting Started

### Set up the Environment

First, you'll need Python 3.7.3. We recommend using `pyenv` to get this (as well as other versions of Python).

Then you'll need Jupyter Notebook. You can find directions on how to get that [here](https://jupyter.org/install). They recommend downloading Anaconda to get it, but we suggest using `pip` to install it if you're already using `pyenv` to manage your python versions.

Then you'll want to clone this repo and `cd` in to it:

```bash
git clone https://github.com/GSA/sagemaker-pilot.git
cd sagemaker-pilot/tokenization
```

We use `pipenv` for a virtual environment. You can find instructions on installing that [here](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv).

Once you've got `pipenv` installed, you can start up the virtual environment using:

```bash
pipenv install
```

This installs the dependencies found in the Pipfile. Next, activate the Pipenv shell:

```bash
pipenv shell
```

This will spawn a new shell subprocess, which can be deactivated by using `exit`.

One of the required packages you just installed is `ipykernel`. We use this to create a kernel that uses our virtual enivronment for the Jupyter Notebook:

```bash
ipython kernel install --user --name=tokenization
```

### Get the labeled training data

You can find the pre-labeled documents [here](https://drive.google.com/drive/folders/1jSlRzeZuKj4RRUrgrjXoVcQXsrMtZfB4). Download them and move them into a new directory named `labeled_fbo_docs`.


### Configure the AWS CLI
The `awscli` python package was included as a dependency, but you still need to configure it using `aws configure`. See this [doc](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration) on how to do that.


### Open the Notebook
At this point, you can start jupyter with `jupyter notebook`. Open `Tokenization.ipynb` and select the kernel (tokenization) that you created a moment ago. 

In the first section of the notebook, you'll create two files in the RecordIO format:  `srt.train` and `srt.validation`.

Then you'll push these files to your S3 bucket. Make sure you adust the name of the bucket to reflect your bucket's name.

## License

This project is licensed under the Creative Commons Zero v1.0 Universal License - see the [LICENSE.md](https://github.com/GSA/sagemaker-pilot/blob/master/LICENSE) file for details.
