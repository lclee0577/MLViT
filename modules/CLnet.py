import torch
import torch.nn as nn
from lightseq.training import LSTransformerEncoderLayer,LSTransformerDecoderLayer

class CLTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(CLTransformerEncoder, self).__init__()
        self.config = config

        embed_dim = config.hidden_size

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(config) for _ in range(config.num_encoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def build_encoder_layer(self, config):
        enc_config = LSTransformerEncoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_imgPad_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
        )
        return LSTransformerEncoderLayer(enc_config)



    def forward(self, x):

        encoder_padding_mask = torch.zeros(x.shape[0:2],dtype=torch.bool)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        x = self.layer_norm(x)
        x = x.transpose(0, 1)

        return x, encoder_padding_mask


class CLTransformerDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super(CLTransformerDecoder, self).__init__()
        self.config = config
        self.use_lang_emb = False
        embed_dim = embed_tokens.config.embedding_dim
        self.embed_tokens = embed_tokens
        self.padding_idx = self.config.padding_idx
        if config.n_langs > 1 :
            self.use_lang_emb = True
            self.lang_embeddings = nn.Embedding(10, embed_dim,padding_idx=config.padding_idx)
            
        self.layers = nn.ModuleList(
            [self.build_decoder_layer(config) for _ in range(config.num_decoder_layer)]
        )
        self.num_layers = len(self.layers)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Linear(
            self.embed_tokens.embeddings.shape[1],
            self.embed_tokens.embeddings.shape[0],
            bias=False,
        )
        self.output_projection.weight = self.embed_tokens.embeddings

    def build_decoder_layer(self, config):
        dec_config = LSTransformerDecoderLayer.get_config(
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            nhead=config.nhead,
            attn_prob_dropout_ratio=config.attn_prob_dropout_ratio,
            activation_dropout_ratio=config.activation_dropout_ratio,
            hidden_dropout_ratio=config.hidden_dropout_ratio,
            pre_layer_norm=config.pre_layer_norm,
            activation_fn=config.activation_fn,
            fp16=config.fp16,
            local_rank=config.local_rank,
            nlayer=config.num_decoder_layer,
        )
        return LSTransformerDecoderLayer(dec_config)

    def forward_embedding(self, trg_tokens, cache=None):
        step = 0
        if cache is not None:
            step = trg_tokens.size(1) - 1
            trg_tokens = trg_tokens[:, -1:]
        x = self.embed_tokens(trg_tokens, step)
        return x

    def forward(self, trg_tokens, encoder_out, encoder_padding_mask,cache=None):
        x = self.forward_embedding(trg_tokens, cache)
        
        if self.use_lang_emb:
            x = x + self.lang_embeddings(trg_tokens[:,0].expand(trg_tokens.size(1),trg_tokens.size(0)).transpose(0, 1))

        if cache == {}:
            for i in range(self.num_layers):
                cache[i] = {}

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache else None
            x = layer(
                x,
                encoder_out,
                encoder_padding_mask,
                layer_cache,
            )

        x = self.layer_norm(x)
        x = self.output_projection(x)
        return x
    
    