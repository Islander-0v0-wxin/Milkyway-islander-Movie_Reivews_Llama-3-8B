# Milkyway-islander-Movie_Reivews_Llama-3-8B
please download from huggingface using this link: https://huggingface.co/Milkyway-islander/Movie_Reivews_Llama-3-8B/tree/main
--

library_name: transformers
tags:
- code
license: llama3
language:
- en
pipeline_tag: text-generation
---

# Model Card for Model ID
model_id = "Milkyway-islander/Movie_Reivews_Llama-3-8B"
<!-- Provide a quick summary of what the model is/does. -->



## Model Details
Input Models input text only.

Output Models generate text and code only.

### Model Description

<!-- Provide a longer summary of what this model is. -->
This model is trained and fine tuned on 1500 movie reviews from IMDB movie review dataset. It aims to generate highly human like movie reviews. 
This is the model card of a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.

- **Developed by:** [Amber Zhan]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [Text Generation]
- **Language(s) (NLP):** [English]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [Llama3-8b]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]


### Direct Use
You can run conversational inference by loading model directly

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             quantization_config=quantization_config, #quantization is optional 
                                             attn_implementation= "flash_attention_2",
                                             force_download=True,
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda")
input_ids = inputs['input_ids']
num_input_tokens = input_ids.shape[1]
attention_mask = inputs['attention_mask']  # Ensure the attention mask is generated

prompt_text = ""

# Generate the response
output = model.generate(
    **inputs,
    max_length=4096 + num_input_tokens,  # Adjust max_length to account for prompt tokens
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(output[0][num_input_tokens:], skip_special_tokens=True)

print(response)

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

[More Information Needed]

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
