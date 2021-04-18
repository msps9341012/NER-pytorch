import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def pad_input(vector):
    max_len = len(max(vector, key = lambda x: len(x))) 
    cust_func = np.vectorize(pyfunc=lambda x: np.pad(array=x, 
                                              pad_width=(0,max_len), 
                                              mode='constant', 
                                              constant_values=(0,0))[:max_len], otypes=[list])
    try:
        return np.stack(cust_func(vector))
    except:
        return vector.astype('int64')


class Perplexity_estimator:
    def __init__(self):
        with torch.no_grad():
            self.scoring_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.scoring_model.cuda()
            self.scoring_model.eval()
            self.scoring_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
    
    def perplexity_score_batch(self, input_ids=None,past_key_values=None,attention_mask=None,
                               token_type_ids=None,position_ids=None,head_mask=None,inputs_embeds=None,
                               encoder_hidden_states=None,encoder_attention_mask=None,labels=None,
                               use_cache=None,output_attentions=None,output_hidden_states=None,
                               return_dict=None):

        transformer_outputs = self.scoring_model.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.scoring_model.lm_head(hidden_states)

        sent_length = attention_mask.sum(dim=1).cpu().detach().numpy()
        sent_length = sent_length-1

        loss = []
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Shift so that tokens < n predict n
            for i in range(lm_logits.shape[0]):
                length = sent_length[i]
                shift_logits_i = shift_logits[i][:length]
                shift_labels_i = shift_labels[i][:length]
                loss.append(loss_fct(shift_logits_i, shift_labels_i))

        return np.exp(torch.stack(loss).cpu().detach().numpy())
        
    def get_perplexity_score(self,input_text,use_batch=True):

        token_data = self.scoring_tokenizer(input_text)
        if use_batch:
            tokenize_inputs = pad_input(np.array(token_data['input_ids'],dtype=object))
            att_masks = pad_input(np.array(token_data['attention_mask'],dtype=object))
        else:
            tokenize_inputs = token_data['input_ids']
            att_masks = token_data['attention_mask']

        tensor_input = torch.tensor(tokenize_inputs).cuda()
        att_masks_tensor= torch.tensor(att_masks).cuda()
        score = self.perplexity_score_batch(input_ids=tensor_input, labels=tensor_input, attention_mask=att_masks_tensor)
        return score
    
    def perplexity_score(self,sentence):
        '''
        sentence: sentence in string
        original version, only support one single sentence per time
        just for validating the result.
        '''
        tokenize_input = self.scoring_tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input]).cuda()
        loss = self.scoring_model(tensor_input, labels = tensor_input)[0]
        return np.exp(loss.cpu().detach().numpy())
