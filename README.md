<img src="https://huggingface.co/aigeek0x0/radiantloom-mixtral-8x7b-fusion/resolve/main/Radiantloom-Mixtral-8x7B-Fusion.png" alt="Radiantloom Mixtral 8X7B Fusion" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

## Radiantloom Mixtral 8X7B Fusion
The [Radiantloom Mixtral 8X7B Fusion](https://huggingface.co/AIGeekLabs/radiantloom-mixtral-8x7b-fusion), a large language model (LLM) developed by AI Geek Labs, features approximately 47 billion parameters and employs a Mixture of Experts (MoE) architecture. With a context length of 4096 tokens, this model is suitable for commercial use.

From vibes-check evaluations, the Radiantloom Mixtral 8X7B Fusion demonstrates exceptional performance in various applications like creative writing, multi-turn conversations, in-context learning through Retrieval Augmented Generation (RAG), and coding tasks. Its out-of-the-box performance already delivers impressive results, particularly in writing tasks. This model produces longer form content and provides detailed explanations of its actions. To maximize its potential, consider implementing instruction tuning and Reinforcement Learning with Human Feedback (RLHF) techniques for further refinement. Alternatively, you can utilize it in its current form.

## Intended Uses & Limitations
The Radiantloom Mixtral 8X7B Fusion is versatile and can be utilized for various text generation tasks such as summarization, chat, coding, question answering, retrieval augmented generation (RAG), role play, and content generation.

While it may not be considered a state-of-the-art generative language model, it demonstrates competitive performance in general tasks when compared to other open and closed-source large language models such as Mixtral 8x7B Instruct v0.1, and GPT 3.5 Turbo.

This model has not undergone fine-tuning for alignment and safety. As a result, it outputs everything it knows, which may include harmful, toxic, and NSFW content. The model is fully uncensored and does not require any special prompts to change its behavior; it comes with an uncensored behavior out of the box. This uncensored nature contributes to the model's compliance with user requests. However, it is strongly advised to implement your own alignment layer before exposing the model as a service. Without such precautions, the model may be highly compliant with any requests, even those that are unethical. Keep in mind that you are responsible for any content generated using this model, so use it responsibly.

## Prompt Template
In order to leverage instruction fine-tuning, your prompt should be surrounded by [INST] and [/INST] tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.

```
<s>[INST] {user_message} [/INST]
```

## Model Usage
You can try it out for free using this [notebook](https://www.kaggle.com/metheaigeek/radiantloom-mixtral-8x7b-fusion).

For more powerful GPU usage and faster inference, you can deploy it on a Runpod GPU instance using our [one-click Runpod template](https://www.runpod.io/console/gpu-secure-cloud?ref=80eh3891&template=ch3txp7g1c) (Our Referral Link. Please consider Supporting). This template provides you with an OpenAI-compatible API endpoint that you can integrate into your existing codebase designed for OpenAI APIs. To learn more about the deployment process and API endpoint, consult the deployment guide provided [here](https://github.com/aigeek0x0/Radiantloom-Mixtral-8X7B-Fusion/blob/main/runpod-deployment-guide.md).


## Inference Code
```python

## install dependencies
## !pip install -q -U git+https://github.com/huggingface/transformers.git
## !pip install -q -U git+https://github.com/huggingface/accelerate.git
## !pip install -q -U bitsandbytes sentencepiece

## load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

model_name = "aigeek0x0/radiantloom-mixtral-8x7b-fusion"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_4bit=True,
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

## inference
prompt = """
<s>[INST] Take a deep breath and think step-by-step.

Write a tale about a time-traveling historian who's determined to witness the most significant events in human history. [/INST]
"""

# tokenize the input string
inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    return_attention_mask=False
)

# generate text using the model
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
outputs = model.generate(**inputs, max_length=2048, streamer=streamer)

# decode and print the output
text = tokenizer.batch_decode(outputs)[0]

```

Here are a few example use cases. if we query the model with

```
<s>[INST] Compose a poem depicting an AI engineer, burning the midnight oil with dedication, to unveil a groundbreaking model to the world. [/INST]

```

the model will return

```
In the wee hours of the night, when shadows dance and whispers speak,
An AI engineer, eyes aglow with dreams to reap,
Burns the midnight oil, fueled by passion's sweet embrace,
To unveil a model, groundbreaking, full of grace.

In the sanctum of his lab, the hallowed space,
Where code and algorithms intertwine in endless race,
He breathes life into silicon, sips caffeine to stay awake,
As the stars above him wink, and planets align for his wake.

With every line of code, a story unfolds,
A tale of connections, patterns, and hidden truths untold,
And as each line is etched, the model evolves,
A living entity, born from the depths of endless folds.

Through sleepless nights and endless days,
The engineer tames the beast, the machine that obeys,
Until the model emerges, ready to take its place,
In the world, where it will change our human race.

From the realm of dreams, to the realm of reality,
A symphony of ones and zeros, birthed from the ether's sea,
A testament to mankind's ceaseless curiosity,
A beacon of hope, a monument to destiny.

So, let the midnight oil burn bright,
Let the stars guide us through this journey's flight,
For within these walls, a revolution lies,
In the heart of a dreamer, an innovator, an AI's prize.

```

and if we query the model with

```
<s>[INST] If it is true that all men are mortal and Socrates is a man, what can be concluded? [/INST]

```

the model will return

```
If it is true that all men are mortal and Socrates is a man, then it can be concluded that Socrates is mortal. The statement "all men are mortal" is a universal statement that applies to every member of the group "men." Since Socrates is a member of that group, he is also subject to the characteristic of mortality.

```

## Evaluations
We are encouraged by the initial assessments conducted using the [LLM-as-a-Judge method](https://arxiv.org/abs/2306.05685), particularly with the utilization of GPT-4. Our ongoing efforts involve a more comprehensive evaluation of the model. We have also entered this model into the HuggingFace Open Leaderboard for assessment. As we progress, we will continue to provide updates on our findings and share the results on this platform.

## Ethical Considerations and Limitations
Radiantloom Mixtral 8X7B Fusion, a powerful AI language model, can produce factually incorrect output and content not suitable for work (NSFW). It should not be relied upon to provide factually accurate information and should be used with caution. Due to the limitations of its pre-trained model and the finetuning datasets, it may generate lewd, biased, or otherwise offensive content. Consequently, developers should conduct thorough safety testing prior to implementing any applications of this model.

## About Radiantloom
Radiantloom trains open-source large language models tailored for specific business tasks such as copilots, email assistance, customer support, and database operations.

Learn more about Radiantloom by visiting our [website](https://radiantloom.com). Follow us on Twitter at [Radiantloom](https://twitter.com/radiantloom) to gain early access to upcoming Radiantloom large language models.

## Want to Support Our Work?

We are passionate about contributing open-source AI models to the broader AI community to accelerate AI development and deployment. If you find our work valuable and would like to contribute to its continued success, consider buying us a few coffees!

Your support goes a long way in helping us cover training costs, fund development tools, and keep the project thriving. Every coffee you buy is a small but meaningful contribution that fuels our dedication to providing a top-notch experience for users like you.

[Buy us a coffee](https://www.buymeacoffee.com/aigeek_) and be a part of our journey towards making this project even more awesome. We appreciate your support!
