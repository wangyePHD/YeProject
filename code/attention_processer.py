import torch



def inject_forward_crossattention(
    self,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
):
    # note self-attention: encoder_hidden_states为None; cross-attention: encoder_hidden_states不为None

    residual = hidden_states
    
    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    
    query = self.to_q(hidden_states)
    key = None
    value = None
    # note define K V self-attention: to_key,to_val, cross-attention：to_global_key,to_global_val     
    if encoder_hidden_states is None: #note self-attention
        encoder_hidden_states = hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
    elif self.norm_cross: #note cross-attention
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
        key = self.to_k_global(encoder_hidden_states)
        value = self.to_v_global(encoder_hidden_states)
    else:
        key = self.to_k_global(encoder_hidden_states)
        value = self.to_v_global(encoder_hidden_states)
        
    
        
    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    attention_probs = self.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states