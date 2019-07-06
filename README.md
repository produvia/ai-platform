# AI Platform

## Background

Artificial intelligence (AI) gig economy is coming. Many industries including Uber, DoorDash, Fiverr built an economy by relying on contingent workers, available on-demand. The great paradox of on-demand services is that they can be easily automated using AI.

- Where do AI systems come from? Research and development (R&D)
- What do AI systems do today? Perform tasks (i.e. provide services)
- What will AI systems be able to do? Automate human tasks (and more)
- What is R&D? Tasks to automate
- Where does AI R&D automation lead? Recursive technology improvement
- Where does that lead? AI services, which includes the service of developing new services

Unautomated AI R&D tasks are human tasks. Automated AI R&D tasks are computer tasks solved using AI and machine learning. Advances in AI will enable incremental speedup and automation of human tasks.

> "The AI explosion will be enabled by AI-driven AI developments. As AI technologies are being recursively improved, new AI services and new AI tasks will be solved."
>
> -- Slava Kurilyak, Founder / CEO at Produvia

## About

![ai-platform](ai-platform.jpg)

At [Produvia](https://produvia.com), we are developing an **AI Platform** which aims to automate AI R&D tasks. Our vision is to create machine learning models to solve various computer science tasks and achieve full AI automation. We list a few examples of AI tasks below:

- semantic segmentation (computer visions)
- machine translation (natural language processing)
- word embeddings (methodology)
- recommendation systems (miscellaneous)
- speech recognition (speech)
- atari games (playing games)
- link prediction (graphs)
- time series classification (time series)
- audio generation (audio)
- visual odometry (robots)
- music information retrieval (music)
- dimensionality reduction (computer code)
- decision making (reasoning)
- knowledge graphs (knowledge base)
- adversarial attack (adversarial)

> "We need to standarize AI solutions by focusing on solving AI R&D tasks while developing machine learning models that are reusable and easily accessible by all."
>
> -- Slava Kurilyak, Founder / CEO at Produvia

**What is our approach?**

We are developing service-centered or task-focused machine learning models. These models, or AI services, solve distinct tasks or functions.

**Proposed Folder Structure:**

We propose to store machine learning models using the following folder structure:

`/tasks/area/task`

For example:

`/tasks/computer-vision/image-classification`

## Installation

To get the entire project up and running locally:

Clone the repo:

```
$ git clone https://github.com/produvia/ai-platform.git
$ cd ai-platform
```

## Running Locally

This is an example of how to run `object detection` on the picture containing zebra:

```
$ cd tasks/computer-vision/object-detection/
$ mlflow run . -P photo_name=zebra.jpg
```

## Project Components

For more information, check out documentation for the different services:

- [/tasks](https://github.com/produvia/ai-platform/tree/master/tasks) - for compilation of AI tasks

## Project Dependencies

![ai-platform-mlflow](ai-platform-mlflow.jpg)

- We use [MLflow](https://github.com/mlflow/mlflow) for model tracking and model deployment. MLflow is an open source platform for machine learning lifecycle. Familiarize yourself with MLflow by going through the following resources:
	1. Review [MLflow examples](https://github.com/mlflow/mlflow/tree/master/examples) hosted on Github.
	2. Watch [MLflow intro video on YouTube](https://www.youtube.com/watch?v=QJW_kkRWAUs) by Matei Zaharia, Co-founder and Chief Technologist at Databricks.
	3. Watch [MLflow videos on YouTube](https://www.youtube.com/playlist?list=PLTPXxbhUt-YVstcW1-OrYoRiAipXRManO)

## Supported Programming Languages

AI Platform supports various programming languages:

<a href="https://www.python.org/"><img src="https://mlflow.org/images/integration-logos/python.png" width="125px" alt="Python" title="Python"></a> <a href="https://www.r-project.org/"><img src="https://mlflow.org/images/integration-logos/r.png" width="125px" alt="R" title="R"></a> <a href="https://en.wikipedia.org/wiki/Java_(programming_language)"><img src="https://mlflow.org/images/integration-logos/java.png" height="80px" alt="Java" title="Java"></a>

## Supported ML Development Frameworks

AI Platform supports various machine learning frameworks and libraries:

<a href="https://tensorflow.org/"><img src="https://mlflow.org/images/integration-logos/tensorflow.png" width="125px" alt="Tensorflow" title="Tensorflow"></a> <a href="https://pytorch.org/"><img src="https://mlflow.org/images/integration-logos/pytorch.png" width="125px" alt="Pytorch" title="Pytorch"></a> <a href="https://keras.io/"><img src="https://mlflow.org/images/integration-logos/keras.png" width="125px" alt="Keras" title="Keras"></a> <a href="https://spark.apache.org/"><img src="https://mlflow.org/images/integration-logos/apache-spark.png" width="125px" alt="Apache Spark" title="Apache Spark"></a> <a href="https://scikit-learn.org/"><img src="https://mlflow.org/images/integration-logos/scikit-learn.png" width="125px" alt="Scikit Learn" title="Scikit Learn"></a> <a href="https://www.h2o.ai/"><img src="https://mlflow.org/images/integration-logos/h2o.png" height="80px" alt="H2O" title="H2O"></a>

## Supported ML Deployment Frameworks

AI Platform supports various machine learning deployment frameworks and libraries:

<a href="https://conda.io"><img src="https://mlflow.org/images/integration-logos/conda.png" width="125px" alt="Conda" title="Conda"></a> <a href="https://www.docker.com/"><img src="https://mlflow.org/images/integration-logos/docker.png" width="125px" alt="Docker" title="Docker"></a> <a href="http://mleap-docs.combust.ml/"><img src="https://mlflow.org/images/integration-logos/mleap.png" width="125px" alt="MLeap" title="Mleap"></a> <a href="https://aws.amazon.com/sagemaker/"><img src="https://mlflow.org/images/integration-logos/sagemaker.jpg" width="125px" alt="SageMaker" title="SageMaker"></a> <a href="https://azure.microsoft.com/en-ca/services/machine-learning-service/"><img src="https://mlflow.org/images/integration-logos/azure-ml.png" width="125px" alt="Azure ML" title="Azure ML"></a> <a href="https://cloud.google.com/"><img src="https://mlflow.org/images/integration-logos/google-cloud.png" height="80px" alt="Google Cloud" title="Google Cloud"></a>

## Common Questions

### What is the difference between AI Platform and MLflow?

AI Platform is an open source platform for automating machine learning models (aka tasks) while MLflow is open source platform for the machine learning lifecycle. AI Platform focuses on automating tasks while MLflow focuses on managing and deploying models. AI Platform focuses on automating model building. MLflow focuses on automating post-modeling. AI Platform is dependent on MLflow. MLflow is not dependent on AI Platform.

AI Platform:

> AI Platform Tasks: Automate computer science tasks using AI, machine learning, deep learning and data science.

MLflow:

> MLflow Tracking: Log parameters, code, and results in machine learning experiments and compare them using an interactive UI.
>
> MLflow Projects: A code packaging format for reproducible runs using Conda and Docker, so you can share your ML code with others.
>
> MLflow Models: A model packaging format and tools that let you easily deploy the same model (from any ML library) to batch and real-time scoring on platforms such as Docker, Apache Spark, Azure ML and AWS SageMaker.

### What is the difference between AutoML and AI Platform?

AutoML aims to "make machine learning more accessible by automatically generating a data analysis pipeline that can include data pre-processing, feature selection, and feature engineering methods along with machine learning methods and parameter settings that are optimized for your data." ([AutoML.info, 2019](http://automl.info/))

At Google, AutoML aims to "make AI accessible to every business". ([Google, 2019](https://cloud.google.com/blog/topics/inside-google-cloud/cloud-automl-making-ai-accessible-every-business))

At [Produvia](https://produvia.com), AI Platform aims to make machine learning models reusable and easily accessible by all.

I agree with you in that requiring S3 distributed storage is not a requirement, but a recommendation. As long we keep an eye on the repository size, using `git-sizer` (https://github.com/github/git-sizer) or similar tools, we'll be in good shape.

We can include a brief note on datasets, such as:

## Datasets

### Uploading Your Own Dataset

Do you have your own data? We recommend that you upload your own dataset onto a public bucket via [AWS S3](https://aws.amazon.com/s3) and include a `LICENSE` file which describes commercial or non-commercial rights.

### Finding an Existing Dataset

Are you looking for a dataset? We recommend that you check out [DataSetList.com](https://www.datasetlist.com/) which includes biggest machine learning datasets for computer vision (CV), natural language processing (NLP), question answer (QA), audio, and medical industries.

## Contributing

### Code

When contributing to the codebase, please create a new feature branch:

```
$ git checkout -b feature/<YOUR_FEATURE_NAME>
```

To push your latest changes to the repo:

```
$ git push origin feature<YOUR_FEATURE_BRANCH>
```

When you are ready to merge your feature branch back into `master`:

1. Ensure you have pushed your latest changes to the origin feature/<FEATURE_BRANCH> branch
2. Submit a pull request to the `master` branch

### Ideas

Do you have an idea on how we can improve **AI Platform**? Email us at [hello@produvia.com](mailto:hello@produvia.com)

## About Produvia

AI Platform was launched by the Produvia team. Since 2013, [Produvia](http://produvia.com) has partnered with companies from all industries to accelerate the adoption of AI and machine learning technologies.