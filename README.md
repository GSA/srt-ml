# sagemaker-pilot
This project pilots AWS SageMaker at GSA. Instead of using one of the trivial examples notebooks provided by Amazon, we create a pipeline transfomer connected to a binary classifier for a text classification challenge. That inference pipeline is then deployed to an endpoint where inferences on new samples can be made via REST API calls.

## Getting Started

### Prerequisites

#### Virtual Environemnt

We use [pipenv](https://github.com/pypa/pipenv) for dependency management. In the root of this repo, install your dependencies with:

```bash
pipenv install
```

Now start a shell with that environment:

```bash
pipenv shell
```

This will spawn a new shell subprocess, which can be deactivated by using `exit`.

One of the required packages you just installed is `ipykernel`. We use this to create a kernel that uses our virtual enivronment for the Jupyter Notebook:

```bash
ipython kernel install --user --name=tokenization
```

#### Get the labeled training data

You can find the pre-labeled documents [here](https://drive.google.com/drive/folders/1jSlRzeZuKj4RRUrgrjXoVcQXsrMtZfB4). Download them and move them into a new directory named `labeled_fbo_docs`.


#### Configure the AWS CLI

The `awscli` python package was included as a dependency, but you still need to configure it using `aws configure`. See this [doc](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration) on how to do that. NOTE: You should have already created an IAM user for this project - as well as the infrastructure - following the steps [here](https://docs.google.com/document/d/1R8JgXL1Pgz67-0d8J_NXQ3TSOZErIJaREr3LqVC5rdI/edit#).

### Open the Notebook

At this point, you can start jupyter with `jupyter notebook`. Open `Tokenization.ipynb` and select the kernel that you created a moment ago. 

From here, follow the steps in the notebook to push these files to your S3 bucket. Make sure you adust the name of the bucket to reflect your bucket's name.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/GSA/sagemaker-pilot/blob/master/.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the Creative Commons Zero v1.0 Universal License - see the [LICENSE.md](https://github.com/GSA/sagemaker-pilot/blob/master/.github/LICENSE) file for details.
