# See It from My Perspective: Diagnosing the Western Cultural Bias of Large Vision-Language Models in Image Understanding (2024)

Amith Ananthram, Elias Stengel-Eskin, Carl Vondrick, Mohit Bansal, Kathleen McKeown

Paper: https://arxiv.org/abs/2406.11665

This codebase allows replication of the experiments in our paper which characterize and root cause Western bias in VLMs.  

It depends on [our fork of the fantastic LLaVA repository](https://anonymous.4open.science/r/mLLaVA) which we have adapted to support using the [Baichuan2](Baichuan2) family of LLMs as the base LLM.  If you're only interested in using our monolingual/bilingual Llama2 and Baichuan2 based VLMs, you only need to clone our LLaVA fork (or you can simply use the models directly [via HuggingFace](https://huggingface.co/papers/2406.11665).

![Teaser image: Are multilingual VLMs actually multicultural?](figures/teaser.png "Are multilingual VLMs actually multicultural?")

## Usage

