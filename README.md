# srt-ml

This project will provide the machine learning component of the Solicitation Review Tool using AWS SageMaker. 

For now, the project includes only model training and deployment functionality, using SageMaker within a private subnet.

Eventually, a combination of AWS Lambda, API Gateway and Congito will be added to make these SageMaker actions callable through a REST API using Oauth2.

## Getting Started

### Create and Configure an AWS Account

First, you need an AWS account with a VPC configured as described [here](https://docs.google.com/document/d/1R8JgXL1Pgz67-0d8J_NXQ3TSOZErIJaREr3LqVC5rdI/edit).

### Local Development

We use [pipenv](https://github.com/pypa/pipenv) for dependency management and to ensure that your local environment matches that of [AWS SageMaker (particularly the sklearn framework)](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/sklearn/README.rst). 

While in the root of this repo, install your dependencies with:

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
ipython kernel install --user --name=srt-ml
```

### Download the Training Data

The SRT utilizes supervised machine learning. You can find 993 pre-labeled documents [here](https://drive.google.com/drive/folders/1jSlRzeZuKj4RRUrgrjXoVcQXsrMtZfB4). Download them and move them into a new directory named `labeled_fbo_docs/`.


### Configure the AWS CLI

The `awscli` python package was included as a dependency, but you still need to configure it using `aws configure`. See this [doc](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration) on how to do that. 

>NOTE: You should have already created an IAM user for this project - as well as the infrastructure - following the linked above.

### Upload the Training Data

At this point, you can start jupyter with `jupyter notebook`. Open `Upload Training Data to S3.ipynb` and select the kernel that you created a moment ago. 

From here, follow the steps in the notebook to push the labeled data up to your S3 bucket. Make sure you adust the name of the bucket to reflect your bucket's name.

## Using SageMaker

You're now ready to use SageMaker. When creating the SageMaker notebook instance, you should have linked this repository. If so, `srt.ipynb` will already be present once you launch the notebook instance.

> NOTE: you can run shell commands, such as `git pull`, within a Jupyter Notebook cell by prepending the command with an exclamation point, e.g. `! git pull`. Doing this will help you keep your SageMaker notebook instance current with the remote repo.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/GSA/sagemaker-pilot/blob/master/.github/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the Creative Commons Zero v1.0 Universal License - see the [LICENSE.md](https://github.com/GSA/sagemaker-pilot/blob/master/.github/LICENSE) file for details.
