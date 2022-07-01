import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook

from collections import defaultdict

class Logit_Lens:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        layer_module_tmp = "transformer.h.{}",
        layer_output_tmp = "h{}_out",
        k = 10,
    ):
        self.model, self.tok = model, tok
        self.layer_module_tmp = layer_module_tmp
        self.layer_output_tmp = layer_output_tmp
        self.k = k

        self.layer_norm_final = nethook.get_module(self.model, 'transformer.ln_f')
        self.embedding_to_word = nethook.get_module(self.model, "lm_head")
        self.layers = [self.layer_module_tmp.format(idx) for idx in range(self.model.config.n_layer)]

    
    def get_V_space_representation(
        self, 
        value_vector, k = 10, perform_layer_norm=True,
        debug = False, delimiter = " ", prefix = ""
    ):
        value_vector = value_vector.clone().reshape(self.model.config.n_embd)
        if(perform_layer_norm):
            value_vector = self.layer_norm_final(value_vector)
        
        v_space = self.embedding_to_word(value_vector)
        confidence = torch.softmax(v_space, dim=0)
        topk = torch.topk(confidence, dim=0, k = k)

        selected_top_tokens = []

        for i in range(k):
            selected_top_tokens.append((
                self.tok.decode(topk.indices[i]), # token
                float(topk.values[i]), # token confidence
                int(topk.indices[i]), # token id
            ))
        if(debug == True):
            print(prefix + " : " + f",{delimiter}".join([f"{tup[0]}: {tup[1]*100}" for tup in selected_top_tokens]))

        return selected_top_tokens


    def generate_next_token(
        self, 
        prompt, tokenized = False, 
        k = None, debug = False, 
        intervention_function=None, intervention_layer = 17,
        with_previous = False,

        past_key_values = None,
    ):
        if(tokenized == False):
            prompt = self.tok([prompt], padding=True, return_tensors="pt").to(
                next(self.model.parameters()).device
            )
            past_key_values = None

        if(k == None):
            k = self.k
        
        trace_layers = self.layers
        if(intervention_function != None):
            # before_intervention = 'transformer.h.{}.ln_2'.format(intervention_layer)
            intervention_layer_mlp_output = 'transformer.h.{}.mlp'.format(intervention_layer)
            trace_layers.append(intervention_layer_mlp_output)

        # print("Tracing")
        with nethook.TraceDict(self.model, trace_layers, edit_output=intervention_function) as traces:
            # model_out = self.model(**prompt)
            # print(model_out)
            model_out = self.model(
                input_ids = prompt['input_ids'],
                attention_mask = prompt['attention_mask'],
                past_key_values = past_key_values,
                use_cache = True,
            )
            past_key_values = model_out.past_key_values
            layer_outputs = traces
        
        output_vectors = list(defaultdict(list))
        v_space_reprs = list(defaultdict(list))

        if(with_previous == True):
            layer = self.layer_module_tmp.format(0)
            interested_token_positions = list(range(layer_outputs[layer].output[0].shape[1]))
        else:
            interested_token_positions = [-1] # only the last position (the `generated` token)

        for position in interested_token_positions:
            to_return = defaultdict(list)
            output_vectors_for_single_token = defaultdict(list)
            for layer_order in range(self.model.config.n_layer):
                layer = self.layer_module_tmp.format(layer_order)
                h_out = self.layer_output_tmp.format(layer_order)
                # print(h_out, ":", layer_outputs[layer].output[0].shape)

                with torch.no_grad():
                    cur_out = layer_outputs[layer].output[0][:, position].clone()
                    output_vectors_for_single_token[h_out] = cur_out.reshape(self.model.config.n_embd)
                    to_return[h_out] = self.get_V_space_representation(
                        value_vector = cur_out, k = k, perform_layer_norm = True,
                        debug = debug, delimiter = " ", prefix = h_out,
                    )
            output_vectors.append(output_vectors_for_single_token)
            v_space_reprs.append(to_return)

        final_layer = self.layer_output_tmp.format(self.model.config.n_layer - 1)        
        generated_tokens = [top_tokens[final_layer][0] for top_tokens in v_space_reprs]            

        return generated_tokens, v_space_reprs, output_vectors, past_key_values


    def generate_argmax_greedy(
        self, 
        prompt, max_out_len = 30, k = None, 
        debug = False, 
        intervention_function=None, intervention_layer = 17,
        with_previous = True
    ):
        generated_tokens = []
        v_space_reprs = []
        output_vectors = []

        original_prompt_tokenized = [self.tok.decode(t) for t in self.tok.encode(prompt)]
        past_key_values = None

        tokenized_prompt = self.tok([prompt], padding=True, return_tensors="pt").to(
                next(self.model.parameters()).device
        )
        input_ids, attention_mask = tokenized_prompt["input_ids"], tokenized_prompt["attention_mask"]
        assert(
            input_ids.size(1) < max_out_len
        ), f"max_out_len({max_out_len}) must be > token len({input_ids.size(1)})"

        with torch.no_grad():
            while(input_ids.size(1) < max_out_len):
                # print(input_ids)
                # print(attention_mask)
                # print("past_key_values", end = " ")
                # if(past_key_values == None):
                #     print("None")
                # else:
                #     print(len(past_key_values))
                #     check_structure_tree(past_key_values[0])
                
                pass_input_ids = input_ids
                pass_attention_mask = attention_mask
                if(past_key_values != None):
                    pass_input_ids = input_ids[:, -1].reshape(1,1)
                    pass_attention_mask = attention_mask[:, -1].reshape(1,1)
                
                next_token, next_v_space_reprs, next_outputs, past_key_values_updated = self.generate_next_token(
                    prompt = {
                        'input_ids': pass_input_ids,
                        'attention_mask': pass_attention_mask
                    },
                    tokenized = True,
                    k = k, debug = debug,
                    intervention_function= intervention_function,
                    with_previous = with_previous,

                    past_key_values = past_key_values
                )
                # print(next_token)
                generated_tokens += next_token
                v_space_reprs += next_v_space_reprs
                output_vectors += next_outputs
                past_key_values = past_key_values_updated

                # prompt += next_token[-1][0]
                with_previous = False
                intervention_function = None

                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(1, 1)], dim = 1)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token[-1][2]]]).to(input_ids.device)], dim = 1)
            

        
        return generated_tokens, v_space_reprs, output_vectors, original_prompt_tokenized