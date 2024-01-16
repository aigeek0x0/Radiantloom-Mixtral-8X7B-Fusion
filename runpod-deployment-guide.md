## How to Deploy Radiantloom Mixtral 8X7B Fusion on Runpod
1. Use [this Runpod template](https://www.runpod.io/console/gpu-secure-cloud?ref=80eh3891&template=ch3txp7g1c) to deploy a Pod using an A100 (80GB) GPU instance.
2. Once the Pod is initiated, connect to it using the web terminal.
3. Install the "transformers" library from the GitHub repository.

```
apt update && sudo apt upgrade
pip install --upgrade pip
apt-get install -y git
pip install -q -U git+https://github.com/huggingface/transformers.git
```
4. Install Flash Attention.
```
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
```

5. Check the Pod logs to ensure that the model has been downloaded, and you see a "connected" message towards the end of the logs. This process may take 20-30 minutes and might require 4-5 Pod restarts. There are issues with model downloads causing it to get stuck, so monitor the RAM usage and logs to determine if it's loading or not.

Once connected, you can call the API at the Radiantloom Mixtral 8X7B Fusion generation endpoint using code compatible with the OpenAI API at the following link.
```
https://{ENTER_YOUR_POD_ID}-8080.proxy.runpod.net
```
Here is a [Google Colab link](https://colab.research.google.com/drive/1jQAx3YbyNdY-NNPabMugwJ3pVZ7S3p2g#scrollTo=dqtCxoB002uy) with inference code.

## Radiantloom Mixtral 8X7B Fusion
The Radiantloom Mixtral 8X7B Fusion, a large language model (LLM) developed by AI Geek Labs, features approximately 47 billion parameters and employs a Mixture of Experts (MoE) architecture. With a context length of 4096 tokens, this model is suitable for commercial use.

From vibes-check evaluations, the Radiantloom Mixtral 8X7B Fusion demonstrates exceptional performance in various applications like creative writing, multi-turn conversations, in-context learning through Retrieval Augmented Generation (RAG), and coding tasks. Its out-of-the-box performance already delivers impressive results, particularly in writing tasks. This model produces longer form content and provides detailed explanations of its actions. To maximize its potential, consider implementing instruction tuning and Reinforcement Learning with Human Feedback (RLHF) techniques for further refinement. Alternatively, you can utilize it in its current form.

[Learn More](https://huggingface.co/AIGeekLabs/radiantloom-mixtral-8x7b-fusion)
