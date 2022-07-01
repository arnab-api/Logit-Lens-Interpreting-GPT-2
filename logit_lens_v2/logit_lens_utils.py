import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np

layer_name_tmp = "h{}_out"
final_layer = 47
initial_layer = 0

def getInfoForPlot__TokenProgress(
    v_space_reprs,
    generated_tokens, original_prompt_tokenized,
    layer_skip = 0, start_idx = 0, end_idx = None,
    must_have_layers = [] # initial layer (0) and final layer (47) will be by default present in must_have_layers
):
    if(end_idx == None):
        end_idx = start_idx + 10
    end_idx = min(end_idx, len(generated_tokens))

    y_layer_names = []
    layer_names = list(v_space_reprs[0].keys())
    layers_tobe_plotted = list(range(0 , len(layer_names) , layer_skip+1))
    if(final_layer not in must_have_layers):
        must_have_layers.append(final_layer)
    for l in must_have_layers:
        if l not in layers_tobe_plotted:
            layers_tobe_plotted.append(l)
    y_layer_names = [layer_name_tmp.format(l) for l in sorted(layers_tobe_plotted)]


    x_tokens = []
    for pos in range(start_idx, end_idx):
        token = generated_tokens[pos][0]
        if(pos < len(original_prompt_tokenized) - 1):
            x_tokens.append(token + f" (*{original_prompt_tokenized[pos+1]})")
        else:
            x_tokens.append(token)
    # print(x_tokens)

    confidence_matrix = []
    token_matrix = []
    distribution_matrix_top_k = []

    for layer in y_layer_names:
        confidence_arr = []
        token_arr = []
        distribution_arr_top_k = []

        for token_order in range(start_idx, min(end_idx, len(generated_tokens))):
            gen_tok = generated_tokens[token_order]

            cur_tok = v_space_reprs[token_order][layer][0]
            confidence_arr.append(round(cur_tok[1]*100, 2))

            if(cur_tok[0] == gen_tok[0]):
                token_arr.append("<b><i>" + cur_tok[0] + "</i></b>")
            else:
                token_arr.append(cur_tok[0])
            
            top_k = v_space_reprs[token_order][layer]
            distribution_arr_top_k.append(
                "<br>   ".join([f"\'{tup[0]}\': {round(tup[1]*100, 6)} " for tup in top_k])
            )

        confidence_matrix.append(confidence_arr)
        token_matrix.append(token_arr)
        distribution_matrix_top_k.append(distribution_arr_top_k)
    
    return y_layer_names, x_tokens, confidence_matrix, token_matrix, distribution_matrix_top_k


def add_rectangle_patches(fig, x, y, marker_color="black", marker_line_width=2):
    dy = [0, 0.5, 0, -0.5]
    dx = [-0.5, 0, 0.5, 0]

    symbol = [142, 141, 142, 141]
    marker_size = [25, 60]*2

    for i in range(4):
        fig.add_trace(
            go.Scatter(mode="markers", x=[x+dx[i]], y=[y+dy[i]], marker_symbol=symbol[i],
                        marker_color= marker_color, 
                        marker_line_width=marker_line_width, marker_size=marker_size[i],
                        hoverinfo= 'skip'
                    )
            )
        fig['data'][-1]['showlegend'] = False


def plotTokenProgress__withConfidence(
    y_layer_names,
    x_tokens,
    confidence_matrix,
    token_matrix,
    distribution_matrix_top_k,
    generated_tokens,
    
    original_prompt, original_prompt_tokenized,
    start_idx,

    colorscale = 'greens',
    patch_color = 'black',
    
    pre_generation_patch_color = 'lightgray',
    pre_generation_patch_line_width = 6,

    intervention_layer = None,
    intervention_token_position = None,
    intervention_patch_color = 'darkred',
    intervention_patch_line_width = 6,
):
    z = confidence_matrix
    x = list(range(len(x_tokens)))
    y = list(range(len(y_layer_names)))
    generation_start_position = len(original_prompt_tokenized)

    # print(x, y)
    # print(len(x), len(y))
    # print(np.array(z).shape)

    z_text = token_matrix

    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, 
        annotation_text=z_text, 
        customdata=np.dstack((token_matrix, confidence_matrix, distribution_matrix_top_k)), 
        colorscale=colorscale
    )
    # fig = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
    # fig.update_traces(text=z_text, texttemplate="%{text}")
    # fig.update_xaxes(side="top")
    fig.update_traces(
        hovertemplate="<br>".join([
        #   "Token: <b>%{x}</b>",
        #   "Layer: %{y}"
          "<b>\'%{customdata[0]}\': %{customdata[1]}</b><br>", 
        #   "Confidence: %{customdata[1]}%",
          "Top_k:<br>   %{customdata[2]}"
        ])+"<extra></extra>"
    )

    # add_rectangle_patches(fig, x = 3, y = 13)
    for token_order in range(len(x_tokens)):
        gen_tok = x_tokens[token_order]
        # print(token_order, " --> ", gen_tok)
        for layer_no in range(len(y_layer_names)):
            cur_tok = token_matrix[layer_no][token_order]
            # print(cur_tok, end = " ")
            # if(cur_tok[0] == gen_tok):
            if("(*" in gen_tok):
                add_rectangle_patches(fig, x = token_order, y = layer_no, marker_color=pre_generation_patch_color, marker_line_width=pre_generation_patch_line_width)
            elif("<b><i>" in cur_tok):
                # print("(OK)", end=" ")
                add_rectangle_patches(fig, x = token_order, y = layer_no, marker_color=patch_color)
        # print()
    
    if(intervention_token_position != None):
        assert(
            intervention_layer != None
        ), "an intervention Layer order must be specified"

        intervention_layer_name = layer_name_tmp.format(intervention_layer)
        intervention_token_position -= start_idx
        if(intervention_token_position > 0):
            add_rectangle_patches(fig, x = intervention_token_position, y = y_layer_names.index(intervention_layer_name), marker_color=intervention_patch_color, marker_line_width=intervention_patch_line_width)



    fig.update_layout(yaxis_range=[-.5,len(y)])
    fig.update_layout(xaxis_range=[-.5,len(x)])

    fig.update_layout(
        autosize=False,
        width=90*len(x) + 200,
        height=35*len(y) + 200
    )

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = x,
            ticktext = x_tokens,
            tickfont = dict(family='Courier New, Monospace', color='darkblue', size=17)
        )
    )

    fig.update_layout(
        yaxis = dict(
            tickmode = 'array',
            tickvals = y,
            ticktext = y_layer_names,
            tickfont = dict(family='verdana', color='firebrick', size=14)
        )
    )

    fig.update_layout(
        title = dict(
            text=f"{original_prompt}" + "<b><i>{}</i></b>".format("".join([token[0] for token in generated_tokens][generation_start_position - 1:start_idx])) + " ___",
            font = dict(
                size=20,
                color='rgb(0,0,0)'
            ),
            y = 0.05
        )
    )

    fig.update_layout( plot_bgcolor='white')

    # fig.show()
    return fig

def generate_token_progress_plot(
    original_prompt, original_prompt_tokenized,
    generated_tokens, v_space_reprs,
    start_idx, end_idx, layer_skip = 2,

    colorscale = 'greens',
    patch_color = 'black',

    pre_generation_patch_color = 'lightgray',
    pre_generation_patch_line_width = 6,

    intervention_layer = None,
    intervention_token_position = None,
    intervention_patch_color = 'darkred',
    intervention_patch_line_width = 6,

    must_have_layers = []
):  
    if(intervention_layer != None):
        if(intervention_layer not in must_have_layers):
            must_have_layers.append(intervention_layer)

    y_layer_names, x_tokens, confidence_matrix, token_matrix, distribution_matrix_top_k = getInfoForPlot__TokenProgress(
        v_space_reprs=v_space_reprs,
        generated_tokens= generated_tokens, original_prompt_tokenized = original_prompt_tokenized,
        layer_skip = layer_skip, start_idx = start_idx, end_idx = end_idx,
        must_have_layers = must_have_layers
    )
    return plotTokenProgress__withConfidence(
        y_layer_names, x_tokens, 
        confidence_matrix, token_matrix, distribution_matrix_top_k,
        generated_tokens,
        original_prompt, original_prompt_tokenized, start_idx,

        colorscale = colorscale,
        patch_color = patch_color,

        pre_generation_patch_color = pre_generation_patch_color,
        pre_generation_patch_line_width = pre_generation_patch_line_width,

        intervention_layer = intervention_layer,
        intervention_token_position = intervention_token_position,
        intervention_patch_color = intervention_patch_color,
        intervention_patch_line_width = intervention_patch_line_width,
    )
    

import torch
import unicodedata
from typing import Optional, List
import numpy as np
import collections
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    top_k: int = 5,
    max_out_len: int = 200,
    arg_max_greedy = False,
    debug = False
):
    inp_tok = tok(prompt, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    # print(inp_tok)
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    # print(past_key_values)
    # print(cur_context)
    # print(input_ids[:, cur_context])
    # print(attention_mask[:, cur_context])

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]

            # print(softmax_out_top_k.shape)
            # print(softmax_out_top_k)
            # print(tk)

            if(arg_max_greedy == False):
                new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
                new_toks = torch.gather(tk, 1, new_tok_indices)

            else:
                new_tok_indices = torch.topk(softmax_out_top_k, dim=1, k=1)
                new_toks = torch.gather(tk, 1, new_tok_indices.indices)

            if(debug == True):
                token_id = new_toks[0][0]
                print(f"{tok.decode([token_id])}[{token_id}] -- {softmax_out[0][token_id]*100}", end=" ")
                print("[", end="")
                for t in tk[0]:
                    # print(t)
                    print(f"{tok.decode(t)}({round(float(softmax_out[0][int(t)]*100), 2)})", end=" ")
                print("]")

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt


import copy

child_last   = "└───"
child_middle = "├───"
space_pre    = "    "
middle_pre   = "│   "
def check_structure_tree(obj, key='#', level=0, level_info = {}, max_depth = 2):
    if(level == max_depth+1):
        return

    if(level > 0):
        for i in range(level-1):
            if(level_info[i] == 'last'):
                print(space_pre, end="")
            else:
                print(middle_pre, end="")
        if(level_info[level-1] == 'last'):
            child_pre = child_last
        else:
            child_pre = child_middle
        print(child_pre, end="")
    
    if(key != '#'):
        print(key, end=": ")
    
    num_elem = ""
    if(isinstance(obj, tuple) or isinstance(obj, list)):
        num_elem = f'[{len(obj)}]'
    print(type(obj), num_elem, end=" ")
    if(type(obj) is torch.Tensor):
        print("[{}]".format(obj.shape))
    else:
        print()
    if(isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, dict)):
        if(isinstance(obj, dict)):
            keys = list(obj.keys())
        else:
            keys = list(range(len(obj)))
        
        for idx in range(len(keys)):
            li = copy.deepcopy(level_info)
            if(idx == len(obj)-1):
                li[level] = 'last'
            else:
                li[level] = 'middle'
            check_structure_tree(obj[keys[idx]], key=keys[idx], level = level + 1, level_info = li, max_depth = max_depth)

    

final_layer = 47
initial_layer = 0

def getInfoForPlot__output_difference(
    output_vector_1, output_vector_2,
    generated_tokens_1 = None, generated_tokens_2 = None,
    layer_skip = 0,
    start_idx = 0,
    end_idx = None,
    must_have_layers = [], # initial layer (0) and final layer (47) will be by default present in must_have_layers

    max_standard = 100,

    layer_name_tmp = "h{}_out"
):
    max_len = min(len(output_vector_1), len(output_vector_2))
    if(end_idx == None):
        end_idx = start_idx + 10
    end_idx = min(end_idx, max_len)

    y_layer_names = []
    layer_names = list(output_vector_1[0].keys())
    layers_tobe_plotted = list(range(0 , len(layer_names) , layer_skip+1))

    if(final_layer not in must_have_layers):
        must_have_layers.append(final_layer)
    for l in must_have_layers:
        if l not in layers_tobe_plotted:
            layers_tobe_plotted.append(l)
    y_layer_names = [layer_name_tmp.format(l) for l in sorted(layers_tobe_plotted)]

    x_tokens = None
    if(generated_tokens_1 != None):
        x_tokens = []
        for i in range(start_idx, end_idx):
            x_tokens.append(f"{generated_tokens_1[i][0]}|{generated_tokens_2[i][0]}")
        x_tokens.append("")

    difference_matrix = []
    for layer in y_layer_names:
        difference_arr = []
        for idx in range(start_idx, end_idx):
            diff = float(torch.norm(output_vector_1[idx][layer] - output_vector_2[idx][layer]))
            difference_arr.append(diff)
        difference_matrix.append(difference_arr)
    difference_matrix = np.array(difference_matrix)
    max_standard = max(difference_matrix.max(), max_standard)

    # print(difference_matrix.shape)
    add = np.array([max_standard]*difference_matrix.shape[0]).reshape(-1,1)
    # print(add.shape)
    difference_matrix = np.concatenate((difference_matrix, add), axis = 1)

    return y_layer_names, x_tokens, difference_matrix


import plotly.figure_factory as ff
def plotDifferenceBetween__OutputVectors(
    y_layer_names, x_tokens, difference_matrix,
    colorscale = "mint"
):
    z = difference_matrix
    x = list(range(difference_matrix.shape[1]))
    y = list(range(difference_matrix.shape[0]))

    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, 
        annotation_text= np.round(difference_matrix, 3),
        colorscale=colorscale,
        hoverinfo= 'skip'
    )

    fig.update_layout(yaxis_range=[-.5,len(y)])
    fig.update_layout(xaxis_range=[-.5,len(x)])

    fig.update_layout(
        autosize=False,
        width=90*(len(x)-1) + 200, height=35*len(y) + 200
    )
    if(x_tokens != None):
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array', tickvals = x, ticktext = x_tokens,
                tickfont = dict(family='Courier New, Monospace', color='darkblue', size=17)
            )
        )
    fig.update_layout(
        yaxis = dict(
            tickmode = 'array', tickvals = y, ticktext = y_layer_names,
            tickfont = dict(family='verdana', color='firebrick', size=14)
        )
    )

    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
        plot_bgcolor='white'
    )
    return fig


def generate_difference_plot(
    output_vector_1, output_vector_2,
    generated_tokens_1 = None, generated_tokens_2 = None,
    layer_skip = 0, start_idx = 0, end_idx = None,
    must_have_layers = [], # initial layer (0) and final layer (47) will be by default present in must_have_layers

    colorscale = "mint", max_standard = 100,
):
    y_layer_names, x_tokens, difference_matrix = getInfoForPlot__output_difference(
        output_vector_1 = output_vector_1, output_vector_2 = output_vector_2,
        generated_tokens_1 = generated_tokens_1, generated_tokens_2 = generated_tokens_2,
        layer_skip = layer_skip, start_idx = start_idx, end_idx = end_idx,
        must_have_layers = must_have_layers,
        max_standard = max_standard,
    )

    return plotDifferenceBetween__OutputVectors(
        y_layer_names= y_layer_names, x_tokens= x_tokens, difference_matrix= difference_matrix,
        colorscale= colorscale, 
    )