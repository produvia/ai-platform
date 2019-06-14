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

## About AI Platform

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

We are using an open-source machine learning lifecycle platform, [MLflow](https://mlflow.org/), to manage, improve and automate AI tasks.

**Proposed Folder Structure:**

We propose to store machine learning models using the following folder structure:

`/tasks/area/task`

For example:

- [/tasks/computer-vision/image-classification](https://github.com/produvia/ai-platform/tree/master/tasks/computer-vision/image-classification)
- [/tasks/natural-language-processing/text-classification](https://github.com/produvia/ai-platform/tree/master/tasks/natural-language-processing/text-classification)

## Get Started

1. Clone this repo and review the `examples` folder.

2. When you want to contribute, solve one specific AI task using [MLflow](https://mlflow.org/) and issue a [new pull request](https://github.com/produvia/ai-platform/pulls).