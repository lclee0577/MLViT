from dataclasses import dataclass
from torch import nn
import torch
from lightseq.training.ops.pytorch.transformer_embedding_layer import LSTransformerEmbeddingLayer
from lightseq.training.ops.pytorch.util import MODEL_ARCH
from modules.CLnet import CLTransformerDecoder, CLTransformerEncoder
from modules.patch_embed import PatchEmbed
from transformers import BertTokenizer


class CLViT(nn.Module):
    def __init__(self, config,device=None):
        super(CLViT, self).__init__()
        self.config = config
        self.num_tokens = 1

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.build_model(self.config,device=device)

    
    def build_model(self, config,embed_layer=PatchEmbed,device=None):
        
        patchDtype = torch.half if config.fp16 else torch.float
        
        self.patch_embed = embed_layer(img_size=(64,192), patch_size=config.patch_size, in_chans=3, embed_dim=config.hidden_size,dtype=patchDtype,device=device)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, config.hidden_size)) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.encoder = self.build_encoder(config)
        
    def build_encoder(self, config):
        return CLTransformerEncoder(config)
    
    def forward(self,img):
        x = self.patch_embed(img)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  
        x = torch.cat((cls_token, x), dim=1)
       
        src_tokens = x + self.pos_embed
        
        encoder_out, _ = self.encoder(src_tokens)
        enShape = encoder_out.shape
        
        encoder_padding_mask = torch.zeros((enShape[1],enShape[0]),dtype=torch.bool)
        return encoder_out,encoder_padding_mask


class CLgpt2(nn.Module):
    def __init__(self, config):
        super(CLgpt2, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        self.tokenizer = BertTokenizer(config.vocLargePath)
        self.sep_id = self.tokenizer.encode(
                self.tokenizer.special_tokens_map["sep_token"], add_special_tokens=False
                )[0]
        self.tokenizerSmall = BertTokenizer(config.vocSmallPath,cls_token='[smCLS]')
        self.tokenizerRev = BertTokenizer(config.vocSmallPath,cls_token='[rvCLS]')
        
        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)

        self.build_model(self.config)
    
    def build_model(self, config):
        
        decoder_embed_tokens = self.build_embedding(config)
        self.decoder = self.build_decoder(config, decoder_embed_tokens)
        

    def build_decoder(self, config, embed_tokens):
        return CLTransformerDecoder(config, embed_tokens)
    
    def build_embedding(self, config):
        emb_config = LSTransformerEmbeddingLayer.get_config(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=config.hidden_size,
            max_batch_tokens=config.max_batch_tokens,
            max_seq_len=config.max_seq_len,
            padding_idx=config.padding_idx,
            dropout=config.hidden_dropout_ratio,
            fp16=config.fp16,
            local_rank=config.local_rank,
        )
        emb = LSTransformerEmbeddingLayer(emb_config)
        return emb
    
    def convertToken(self,labels,size='large'):
        if size == 'large':
            trg_tokens = self.tokenizer.batch_encode_plus(labels, padding=True, return_tensors="pt")
            trg_tokens = trg_tokens["input_ids"].to(self.device)
        elif size == 'small':
            trg_tokens = self.tokenizerSmall.batch_encode_plus(labels, padding=True, return_tensors="pt")
            trg_tokens = trg_tokens["input_ids"].to(self.device)
        elif size == 'rev':
            revFunc = lambda x:x[::-1]
            revText = list(map(revFunc,labels))
            trg_tokens = self.tokenizerRev.batch_encode_plus(revText, padding=True, return_tensors="pt")
            trg_tokens = trg_tokens["input_ids"].to(self.device)
            
        target = trg_tokens.clone()[:, 1:]
        trg_tokens = trg_tokens[:, :-1]

        return trg_tokens, target

    def forward(self, trg_tokens ,encoder_out, encoder_padding_mask,returnAccu=False,size='large'):
        
        decoder_out = self.decoder(trg_tokens, encoder_out, encoder_padding_mask)

        accu = -1
        if returnAccu:
            outputLabel = torch.reshape(torch.argmax(
                decoder_out, dim=-1), (trg_tokens.size(0), -1))

            mask = torch.cumsum(torch.eq(outputLabel, self.sep_id).int(), dim=1)
            predict_tokens = outputLabel.masked_fill(mask > 0, 0)


            temp = predict_tokens[:, :-1]

            temp2 = trg_tokens[:, 1:]
            mask2 = torch.cumsum(torch.eq(temp2, self.sep_id).int(), dim=1)
            temp2 = temp2.masked_fill(mask2 > 0, 0)

            result = temp.eq(temp2)
            accu = (result.all(dim=1).float().mean())
        return accu,decoder_out

    
    
    def predict(self,encoder_out, encoder_padding_mask,trgLen,size='large'):
        
        predict_tokens = torch.ones((encoder_out.size(1),1),dtype=torch.int64,device= self.device)#encoder_out.size(1) batch size
        if size == 'small':
            predict_tokens = predict_tokens * 4 #[auxCLS] <=> 4 ,[CLS] <=> 1 tokenizer.encode(tokenizer.cls_token)[0]
        elif size == 'rev':
            predict_tokens = predict_tokens * 5 #[auxCLS] <=> 4 ,[CLS] <=> 1 tokenizer.encode(tokenizer.cls_token)[0]
        
        for _ in range(trgLen):
            output = self.decoder(predict_tokens, encoder_out, encoder_padding_mask)
            outputLabel = torch.reshape(torch.argmax(output, dim=-1), (predict_tokens.size(0), -1))
            predict_tokens = torch.cat([predict_tokens, outputLabel[:,-1:]], dim=-1)
        
        value,index = torch.softmax(output,dim=-1).max(-1)

        mask = torch.cumsum(torch.eq(predict_tokens, self.sep_id).int(), dim=1)
        predict_tokens = predict_tokens.masked_fill(mask > 0, self.sep_id)
        
        confArray =  torch.div(value,0.95).cumprod(-1)#0.95 label smooth
        _,end = torch.cumsum( torch.ne(predict_tokens[:,1:], 2).int(), dim = -1 ).max( -1,keepdim=True )
        confidence = torch.gather(confArray,1,end).squeeze(-1) 

        if size=='large':
            predict_text = self.tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)
        elif size == 'rev':
            revText = self.tokenizerRev.batch_decode(predict_tokens, skip_special_tokens=True)
            revFunc = lambda x:x[::-1]
            predict_text = list(map(revFunc,revText))
            
        else:
            predict_text = self.tokenizerSmall.batch_decode(predict_tokens, skip_special_tokens=True)
        return predict_text,confidence

class CLstr(nn.Module):
    def __init__(self, config,device=None):
        super(CLstr, self).__init__()
        self.config = config
        self.vit = CLViT(config,device)
        self.gpt2VocLarge = CLgpt2(config)

        
    @staticmethod
    def get_config(**kwargs):
        @dataclass
        class Config:
            max_batch_tokens: int  # max batch token numbers
            max_seq_len: int  # max sequence length
            max_imgPad_len :int #max image patch len
            n_langs:int
            vocSmallPath :str
            vocLargePath:str
            padding_idx: int  # index of padding token
            num_encoder_layer: int  # number of encoder layer
            num_decoder_layer: int  # number of decoder layer
            hidden_size: int  # size of transformer hidden layers
            intermediate_size: int  # size of ffn inner size
            nhead: int  # number of heads in attention
            attn_prob_dropout_ratio: float  # attention score dropout ratio
            activation_dropout_ratio: float  # ffn activation dropout ratio
            hidden_dropout_ratio: float  # dropout ration before residual
            pre_layer_norm: bool  # pre layer norm or post
            activation_fn: str  # relu or gelu
            fp16: bool  # fp16 presion
            local_rank: int  # rank in local node
            patch_size:int

        if "model" in kwargs:
            if kwargs["model"] not in MODEL_ARCH:
                raise ValueError("{} architecture is not supported.")
            MODEL_ARCH[kwargs["model"]](kwargs)
            del kwargs["model"]

        return Config(**kwargs)
    
    def forward(self, img, trg_tokens=None,returnAccu = False):
        encoder_out, encoder_padding_mask = self.vit(img)    
        accu,decoderOut = self.gpt2VocLarge(trg_tokens,encoder_out,encoder_padding_mask,returnAccu)
        
        return accu,decoderOut


    
    def calcLabelLen(self,label):
        smallLen = self.gpt2VocLarge.convertToken(label,size='small')[0].size(1)
        largeLen = self.gpt2VocLarge.convertToken(label)[0].size(1)
        return smallLen,largeLen
    
    def predict(self,img,trg_len=27,size='large'):
        encoder_out, encoder_padding_mask = self.vit(img)    
        prd_txt,confidence = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_len,size=size)   
         
        return prd_txt,confidence
        
    def predictWithDual(self,img,trg_len1=32,trg_len2=32):
        encoder_out, encoder_padding_mask = self.vit(img)
        prd_txtS,confidenceS = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_len1,size='small')
        prd_txtL,confidenceL = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_len2,size='large')
        
        prd_txt = []
        for txtS,confS,txtL,confL in zip(prd_txtS,confidenceS,prd_txtL,confidenceL):
            
            if confS > confL:
                prd_txt.append(txtS)
            else:
                prd_txt.append(txtL)

        return prd_txt
    
    def predictWithTri(self,img,trg_lenS=27,trg_lenL=27,thre=0.4,ratio=1.2):
        encoder_out, encoder_padding_mask = self.vit(img)
    
        prd_txtS,confidenceS = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_lenS,size='small')
        prd_txtL,confidenceL = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_lenL,size='large')
        prd_txtR,confidenceR = self.gpt2VocLarge.predict(encoder_out, encoder_padding_mask,trg_lenL,size='rev')
        
        prd_txt = []
        for txtS,confS,txtL,confL,txtR,confR in zip(prd_txtS,confidenceS,prd_txtL,confidenceL,prd_txtR,confidenceR):
            
            if txtS == txtR:
                if confL>thre:
                    if confS/confL>ratio:
                        prd_txt.append(txtS)
                        continue
                    else:
                        prd_txt.append(txtL)
                        continue
                      
            confList = [confS,confL,confR ]
            txtList = [txtS,txtL,txtR]    
            prd_txt.append(txtList[confList.index(max(confList))])
        return prd_txt
        
        

    
    