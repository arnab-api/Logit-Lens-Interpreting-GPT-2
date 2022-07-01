# A Logit Lens implementation for the `gpt2-xl` model
## original idea from [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

![Alt text](Figure/demo.png?raw=true "Demo")

The bold grey bars represents the vocabulary space representations **before** the actual generation starts. If the `prompt` was `"Eiffel Tower is located in the"` the next generated token would have been `heart`. Tokens in parenthesis `(* <token>)` is the original token at the prompt.<br/>
The black patches on some cells represents that at that layer the language model was already confident of which token will come next.<br/>
The tooltip shows the top 10 tokens with the highest logit values. 

## Installation
This implementation was tested with the conda environment dump at `environment.yml`.<br/>
`conda env create -f environment.yml`<br/>
However, you should not need to install *all* of the modules in my `conda` working environment. I will try to filter only the packages needed to run the logit lens at a later time.

### !warning
The visualizations are implementd with `plotly` and at this time **GitHub** does not support `plotly` figures in notebook. So, you will not see any figures if you simply click on the notebooks. You will have to clone the repo and install the dependencies to be able to see the figures. 