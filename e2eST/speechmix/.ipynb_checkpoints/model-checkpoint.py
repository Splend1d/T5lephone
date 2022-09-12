import math

from torch import nn
from transformers import AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2FeatureExtractor, AutoModelForCTC, AutoProcessor, T5ForConditionalGeneration, \
    AutoModelForPreTraining, Wav2Vec2Model, PreTrainedModel, PretrainedConfig
from transformers import AutoModelForSeq2SeqLM, SpeechEncoderDecoderModel, AutoTokenizer, \
    Wav2Vec2FeatureExtractor, HubertModel, UniSpeechSatModel, Wav2Vec2Model, PreTrainedModel, PretrainedConfig, \
    AutoConfig
import torch
#import s3prl.hub as hub
import torch.nn.functional as F
import copy


def handle_decoder_input_none(decoder_config, batch=1, device='cpu'):
    return torch.tensor([[decoder_config.decoder_start_token_id]] * batch).to(device)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class EncoderOutput():
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class SpeechMixConfig(PretrainedConfig):
    model_type = "speechmix"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert ("encoder" in kwargs and "decoder" in kwargs), \
            "Config has to be initialized with encoder and decoder config"
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_configs(cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig,
                     **kwargs) -> PretrainedConfig:
        encoder_config = AutoConfig.from_pretrained(encoder_config)
        decoder_config = AutoConfig.from_pretrained(decoder_config)

        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["decoder"] = self.decoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class SpeechMixED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(SpeechMixED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(speech_model_config, nlp_model_config)
        self.model.config.decoder_start_token_id = self.model.config.decoder.decoder_start_token_id
        self.model.config.pad_token_id = self.model.config.decoder.pad_token_id
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(speech_model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.model.freeze_feature_encoder()
        if fixed_parameters:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any([k in name for k in fixed_except]):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

    def forward(self, input_values, attention_mask=None, decoder_input_ids=None, labels=None):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.model.config.decoder, device=self.device)
        outputs = self.model(input_values=input_values, attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict = {'logits': torch.argmax(outputs['logits'], -1)}
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict

class SpeechMixEEDT5(PreTrainedModel):
    main_input_name = "input_values"
    def __init__(self, speech_model_config, nlp_model_config, share_layer_ratio=0,
                 down_scale=8, weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], nlp_model_decoder_only = False,
                 **kwargs
                 ):
        config = SpeechMixConfig.from_configs(speech_model_config, nlp_model_config)
        super(SpeechMixEEDT5, self).__init__(config)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(nlp_model_decoder_only)
        #s()
        #self.encoder_model = getattr(hub, speech_model_config)().to(self.device)
        
        self.encoder_processor = AutoProcessor.from_pretrained(speech_model_config, sampling_rate = 16000)#"facebook/wav2vec2-large-960h-lv60-self"
        self.encoder_model = Wav2Vec2Model.from_pretrained(speech_model_config)#"facebook/wav2vec2-large-960h-lv60-self"
        #print(self.encoder_model.config)
        #s()
        self.nlp_model_decoder_only = nlp_model_decoder_only
        if nlp_model_decoder_only:
            print("nlp decoder only")
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config)
        #self.t5encoder = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).encoder
        #self.encoder_model.attention_dropout = 0.3
        #self.decoder_model.dropout= 0.3
        #print(self.decoder_model)
        #s()
        if "mt5" in nlp_model_config:
            self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.decode_max_len = 256 if "byt" in nlp_model_config else 64
        self.weighted_sum = weighted_sum
        
        num_nlp_encoder_layers = 0
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.layers)
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.block)

        # print("Before layer sharing num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers))
        # remove_layers = int(
        #     len(self.encoder_model.model.encoder.layers) * share_layer_ratio) if share_layer_ratio != 0 else 0
        # self.encoder_model.model.encoder.layers = self.encoder_model.model.encoder.layers[
        #                                           :len(self.encoder_model.model.encoder.layers) - remove_layers]
        # self.num_speech_encoder_layers = len(self.encoder_model.model.encoder.layers)
        # print("After layer sharing ",
        #       "num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers),
        #       "num_nlp_encoder_layers", num_nlp_encoder_layers,
        #       "share_layer_ratio", share_layer_ratio,
        #       "remove_layers", remove_layers, )
        self.num_speech_encoder_layers = self.encoder_model.config.num_hidden_layers
        # Downsample
        self.downsize = down_scale
        self.downloop = int(math.log(self.downsize, 2))


        if self.downsize > 1:
            self.length_adapters = nn.Sequential(
                *[nn.Conv1d(in_channels=self.encoder_model.config.output_hidden_size,
                            out_channels=self.encoder_model.config.output_hidden_size,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop)])#.to(self.device)
        else:
            self.length_adapters = nn.Sequential(nn.Identity())#.to(self.device)

        self.weights_sum = nn.Parameter(torch.zeros(self.num_speech_encoder_layers))#.to(self.device)
        #if self.encoder_model.config.output_hidden_size != self.decoder_model.config.hidden_size:
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.config.output_hidden_size,
                                     self.decoder_model.config.hidden_size)#.to(self.device)
        #else:
        #    self.enc_to_dec_proj = nn.Sequential(nn.Identity())#.to(self.device)
        self.custom_modules(**kwargs)

        num_e_params_before = sum(param.numel() for param in self.encoder_model.parameters() if param.requires_grad)
        num_d_params_before = sum(param.numel() for param in self.decoder_model.parameters() if param.requires_grad)
        

        if fixed_parameters:
            pass
            #self.encoder_model.eval()
            #self.decoder_model.eval()
            #for xcoder in [self.encoder_model.named_parameters]:
            #    for name, param in xcoder():
            #        #print("speech",name)
            #        if param.requires_grad:
            #            if any([k in name for k in {"attention","layer_norm"}]):
            #                for k in {"attention","layer_norm"}:
            #                    if k in name:
            #                        #print(k, "in name trainable")
            #                        break
            #                param.requires_grad = True
            #            else:
            #                param.requires_grad = False
            #for xcoder in [self.decoder_model.named_parameters]:
            #    for name, param in xcoder():
            #        #print("text",name)
            #        if param.requires_grad:
            #            if any([k in name for k in {"Attention","layer_norm"}]):
            #                for k in {"Attention","layer_norm"}:
            #                    if k in name:
            #                        #print(k, "in name trainable")
            #                        break
            #                param.requires_grad = True
            #            else:
            #                param.requires_grad = False
                
        print(self.encoder_model.encoder.layers[23].feed_forward.output_dense.weight.requires_grad)
        #s()

        num_e_params = sum(param.numel() for param in self.encoder_model.parameters() if param.requires_grad)
        num_d_params = sum(param.numel() for param in self.decoder_model.parameters() if param.requires_grad)
        print(f"trainable_parameters_before_fix-speech:{num_e_params_before}, text:{num_d_params_before}")
        print(f"trainable_parameters_after_fix-speech:{num_e_params}, text:{num_d_params}")
        #s()

        list_no_grad = []
        list_grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                list_grad.append(name)
            else:
                list_no_grad.append(name)

        #self.speech_encoder_layer = len(self.encoder_model.config.num_hidden_layers)
        self.nlp_emb = self.decoder_model.get_input_embeddings()
        self.nlp_encoder_layer = num_nlp_encoder_layers
        self.list_grad = list_grad
        self.list_no_grad = list_no_grad
        self.init = False

    def custom_modules(self, **kwargs):
        return None

    def cal_loss(self, inputs_embeds=None,  attention_mask=None, decoder_input_ids=None,
                 labels=None, decoder_attention_mask=None):
        if inputs_embeds is not None:
            #print(inputs_embeds.shape)
            output = self.decoder_model(input_ids = None, inputs_embeds=inputs_embeds,
                                        decoder_attention_mask=decoder_attention_mask, labels=labels)
            #print("output",output)
        return output
    
    def cal_loss_decoder_only(self, encoder_hidden_states=None,  attention_mask=None, decoder_input_ids=None,
                 labels=None, decoder_attention_mask=None):
        if encoder_hidden_states is not None:
            #print(inputs_embeds.shape)
            output = self.decoder_model(input_ids = None, 
                                        encoder_outputs=(encoder_hidden_states,),
                                        decoder_attention_mask=decoder_attention_mask, 
                                        labels=labels)
            #print("output",output)
        return output

    def forward(self, input_values, decoder_text_prompt_ids=None, labels=None, decoder_attention_mask=None, return_model_detail=False):
        #print(len(speech_input))
        #s()
        
        return_dict = {}
        #num_e_params = sum(param.numel() for param in self.encoder_model.parameters() if param.requires_grad)
        #num_d_params = sum(param.numel() for param in self.decoder_model.parameters() if param.requires_grad)
        #print(f"trainable_parameters_before_fix-speech:{num_e_params_before}, text:{num_d_params_before}")
        #print(f"trainable_parameters_after_fix-speech:{num_e_params}, text:{num_d_params}")
        #input_values = speech_input.to(self.device)
        #input_values = input_values.to(self.device)
        
        #print(self.decoder_model.encoder.block[11].layer[1].DenseReluDense.wi_1.weight)
        encoder_outputs = self.encoder_model(input_values)
        #print(encoder_outputs)
        inputs_embeds = encoder_outputs['last_hidden_state']
        #print("i",inputs_embeds.shape)
        
        if self.weighted_sum:
            # weighted sum
            stacked_feature = torch.stack(encoder_outputs['hidden_states'], dim=0)
            _, *origin_shape = stacked_feature.shape
            stacked_feature = stacked_feature.view(self.num_speech_encoder_layers, -1)
            norm_weights = F.softmax(self.weights_sum, dim=-1)
            if return_model_detail:
                return_dict['weighted_sum'] = norm_weights
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            inputs_embeds = weighted_feature.view(*origin_shape)
        if return_model_detail:
            return_dict['shape_before_length_adapter'] = inputs_embeds.shape
        #if str(inputs_embeds.device) == "cuda:0" or str(inputs_embeds.device) == "cuda:1":
            #print(inputs_embeds[0][0][:10])
            #print(self.encoder_model.encoder.layers[23].feed_forward.output_dense.weight)
            #print(self.enc_to_dec_proj.weight[0][:10])
        #print(inputs_embeds.device)
        inputs_embeds = self.length_adapters(inputs_embeds.transpose(1, 2)).transpose(1, 2)


        if return_model_detail:
            return_dict['shape_before_enc_dec_projector'] = inputs_embeds.shape
        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)
        if return_model_detail:
            return_dict['shape_after_enc_dec_projector'] = inputs_embeds.shape
        if decoder_text_prompt_ids is not None and not self.nlp_model_decoder_only:
            decoder_ttext_prompt_emds = self.nlp_emb(
                 decoder_text_prompt_ids)
            #print(text_prompt_emds.shape)
            #print(inputs_embeds.shape)
            inputs_embeds = torch.cat((decoder_ttext_prompt_emds, inputs_embeds), 1)
            print("cat")
        
        if self.nlp_model_decoder_only:
            outputs = self.cal_loss_decoder_only(encoder_hidden_states=inputs_embeds, labels=labels, decoder_attention_mask=decoder_attention_mask )
        else:
            print("here")
            outputs = self.cal_loss(inputs_embeds=inputs_embeds, labels=labels, decoder_attention_mask=decoder_attention_mask )

        return_dict['logits'] = torch.argmax(outputs['logits'], -1)
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        #print(self.encoder_model.encoder.layers[23].feed_forward.output_dense.weight)
        #print(self.encoder_model.encoder.layers[23].feed_forward.output_dense.weight.requires_grad)
        
        return return_dict

class SpeechMixEEDT52(SpeechMixEEDT5):
    def __init__(self, speech_model_config, nlp_model_config, share_layer_ratio=0,
                 down_scale=8, weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(SpeechMixEEDT52, self).__init__(speech_model_config, nlp_model_config, share_layer_ratio,
                 down_scale, weighted_sum, fixed_parameters, fixed_except)

class SpeechMixEEDT5eval(SpeechMixEEDT5):
    def __init__(self, speech_model_config, nlp_model_config, share_layer_ratio=0,
                 down_scale=8, weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], 
                 nlp_model_decoder_only = False, **kwargs):
        super(SpeechMixEEDT5eval, self).__init__(speech_model_config, nlp_model_config, share_layer_ratio,
                 down_scale, weighted_sum, fixed_parameters, fixed_except, nlp_model_decoder_only)
        #print(self.nlp_model_decoder_only)
        #print(nlp_model_decoder_only)
        #s()
        self.encoder_output_sample = self.decoder_model.encoder(inputs_embeds = torch.FloatTensor(torch.zeros(1,1,self.decoder_model.config.hidden_size)), return_dict = True)

    def forward(self, speech_input, decoder_text_prompt=None,decoder_text_prompt_ids=None, labels=None, decoder_attention_mask=None, return_model_detail=False):
        return_dict = {}
        #input_values = self.encoder_processor(speech_input, return_tensors="pt", padding="longest", sampling_rate = 16000).input_values
        #input_values = input_values.to(self.device)
        #print(speech_input.shape)
        input_values = speech_input.to(self.device)
        
        encoder_outputs = self.encoder_model(input_values)
        #print(encoder_outputs)
        inputs_embeds = encoder_outputs['last_hidden_state'].to(self.device)
        print(inputs_embeds[0][0])
        #print(input_values[0])
        if self.weighted_sum:
            # weighted sum
            stacked_feature = torch.stack(encoder_outputs['hidden_states'], dim=0)
            _, *origin_shape = stacked_feature.shape
            stacked_feature = stacked_feature.view(self.num_speech_encoder_layers, -1)
            norm_weights = F.softmax(self.weights_sum, dim=-1)
            if return_model_detail:
                return_dict['weighted_sum'] = norm_weights
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            inputs_embeds = weighted_feature.view(*origin_shape)
        if return_model_detail:
            return_dict['shape_before_length_adapter'] = inputs_embeds.shape
        inputs_embeds = self.length_adapters(inputs_embeds.transpose(1, 2)).transpose(1, 2)


        if return_model_detail:
            return_dict['shape_before_enc_dec_projector'] = inputs_embeds.shape
        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)
        if return_model_detail:
            return_dict['shape_after_enc_dec_projector'] = inputs_embeds.shape
        if decoder_text_prompt_ids is not None and not self.nlp_model_decoder_only:
            print("cat")
            text_prompt_emds = self.nlp_emb(decoder_text_prompt_ids.to(self.device))
            #print(text_prompt_emds.shape)
            #print(inputs_embeds.shape)
            inputs_embeds = torch.cat((text_prompt_emds, inputs_embeds), 1)
        
        if labels is not None:
            if self.nlp_model_decoder_only:
                outputs = self.cal_loss_decoder_only(encoder_hidden_states=inputs_embeds, labels=labels, decoder_attention_mask=decoder_attention_mask )
            else:
                outputs = self.cal_loss(inputs_embeds=inputs_embeds, labels=labels, decoder_attention_mask=decoder_attention_mask )
        else:
            outputs = {}
        if self.nlp_model_decoder_only:
            print("here")
            
            #print(ret)
            self.encoder_output_sample.last_hidden_state = inputs_embeds
            
            generated_ids = self.decoder_model.generate(
                encoder_outputs = self.encoder_output_sample , 
                max_length=self.decode_max_len,
                num_beams=10,
            )
        else:
            generated_ids = self.decoder_model.generate(
                inputs_embeds = inputs_embeds, 
                max_length=self.decode_max_len,
                num_beams=10,
            )
        #print(generated_ids.shape)
        #generated_ids = torch.nn.functional.pad(generated_ids,(generated_ids.shape[0],self.decode_max_len-generated_ids.shape[1]),value = -100)
        #print(generated_ids.shape)
        return_dict['logits'] = generated_ids
        if 'loss' in outputs:
           return_dict['loss'] = outputs['loss']
        return return_dict

class SpeechMixEED(nn.Module):
    def __init__(self, speech_model_config, nlp_model_config, share_layer_ratio=0,
                 down_scale=8, weighted_sum=False,
                 fixed_parameters=False,
                 fixed_except=["layer_norm", "encoder_attn", 'enc_to_dec_proj', 'length_adapter',
                               "layernorm_embedding", 'attention', 'encoder'], **kwargs):
        super(SpeechMixEED, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_model = getattr(hub, speech_model_config)().to(self.device)
        self.decoder_model = AutoModelForSeq2SeqLM.from_pretrained(nlp_model_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(nlp_model_config)
        self.weighted_sum = weighted_sum

        num_nlp_encoder_layers = 0
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.layers)
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            num_nlp_encoder_layers = len(self.decoder_model.base_model.encoder.block)

        print("Before layer sharing num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers))
        remove_layers = int(
            len(self.encoder_model.model.encoder.layers) * share_layer_ratio) if share_layer_ratio != 0 else 0
        self.encoder_model.model.encoder.layers = self.encoder_model.model.encoder.layers[
                                                  :len(self.encoder_model.model.encoder.layers) - remove_layers]
        self.num_speech_encoder_layers = len(self.encoder_model.model.encoder.layers)
        print("After layer sharing ",
              "num_speech_encoder_layers", len(self.encoder_model.model.encoder.layers),
              "num_nlp_encoder_layers", num_nlp_encoder_layers,
              "share_layer_ratio", share_layer_ratio,
              "remove_layers", remove_layers, )

        # Downsample
        self.downsize = down_scale
        self.downloop = int(math.log(self.downsize, 2))
        if self.downsize > 1:
            self.length_adapters = nn.Sequential(
                *[nn.Conv1d(in_channels=self.encoder_model.model.final_proj.in_features,
                            out_channels=self.encoder_model.model.final_proj.in_features,
                            kernel_size=2,
                            stride=2) for _ in range(self.downloop)]).to(self.device)
        else:
            self.length_adapters = nn.Sequential(nn.Identity()).to(self.device)

        self.weights_sum = nn.Parameter(torch.zeros(self.num_speech_encoder_layers)).to(self.device)
        self.enc_to_dec_proj = nn.Linear(self.encoder_model.model.final_proj.in_features,
                                         self.decoder_model.config.hidden_size).to(self.device)
        self.custom_modules(**kwargs)
        if fixed_parameters:
            self.encoder_model.eval()
            self.decoder_model.eval()
            for xcoder in [self.encoder_model.named_parameters, self.decoder_model.named_parameters]:
                for name, param in xcoder():
                    if param.requires_grad:
                        if any([k in name for k in fixed_except]):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

        list_no_grad = []
        list_grad = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                list_grad.append(name)
            else:
                list_no_grad.append(name)

        self.speech_encoder_layer = len(self.encoder_model.model.encoder.layers)
        self.nlp_emb = self.decoder_model.get_input_embeddings()
        self.nlp_encoder_layer = num_nlp_encoder_layers
        self.list_grad = list_grad
        self.list_no_grad = list_no_grad

    def custom_modules(self, **kwargs):
        return None

    def cal_loss(self, inputs_embeds=None,  attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if inputs_embeds is not None:
            output = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids, labels=labels)
        return output

    def forward(self, input_values, input_text_prompt=None,decoder_input_ids=None, labels=None,
                return_model_detail=False):
        if decoder_input_ids is None and labels is None:
            decoder_input_ids = handle_decoder_input_none(self.decoder_model.config, len(input_values),
                                                          device=self.device)
        elif decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(labels, self.decoder_model.config.pad_token_id,
                                                   self.decoder_model.config.decoder_start_token_id)
        return_dict = {}
        encoder_outputs = self.encoder_model(input_values)
        inputs_embeds = encoder_outputs['last_hidden_state'].to(self.device)
        if self.weighted_sum:
            # weighted sum
            stacked_feature = torch.stack(encoder_outputs['hidden_states'], dim=0)
            _, *origin_shape = stacked_feature.shape
            stacked_feature = stacked_feature.view(self.num_speech_encoder_layers, -1)
            norm_weights = F.softmax(self.weights_sum, dim=-1)
            if return_model_detail:
                return_dict['weighted_sum'] = norm_weights
            weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
            inputs_embeds = weighted_feature.view(*origin_shape)
        if return_model_detail:
            return_dict['shape_before_length_adapter'] = inputs_embeds.shape
        inputs_embeds = self.length_adapters(inputs_embeds.transpose(1, 2)).transpose(1, 2)
        if return_model_detail:
            return_dict['shape_before_enc_dec_projector'] = inputs_embeds.shape
        inputs_embeds = self.enc_to_dec_proj(inputs_embeds)
        if return_model_detail:
            return_dict['shape_after_enc_dec_projector'] = inputs_embeds.shape
        if input_text_prompt is not None:
            text_prompt = self.nlp_emb(
                self.tokenizer(input_text_prompt, return_tensors='pt')['input_ids'].to(self.device))
            inputs_embeds = torch.cat((text_prompt, inputs_embeds), 1)
        outputs = self.cal_loss(inputs_embeds=inputs_embeds,
                                decoder_input_ids=decoder_input_ids, labels=labels)
        return_dict['logits'] = torch.argmax(outputs['logits'], -1)
        if 'loss' in outputs:
            return_dict['loss'] = outputs['loss']
        return return_dict


class SpeechMixFixed(SpeechMixEED):

    def custom_modules(self, fixed_speech=False, fixed_nlp=True, **kwargs):
        print(fixed_speech, fixed_nlp, kwargs)
        self.encoder_model.eval()
        self.decoder_model.eval()
        if fixed_speech:
            for name, param in self.encoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
        if fixed_nlp:
            for name, param in self.decoder_model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False


class SpeechMixAdapter(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()
        if hasattr(self.decoder_model.base_model.encoder, 'layers'):
            decoder_stack = [self.decoder_model.base_model.encoder.layers,
                             self.decoder_model.base_model.decoder.layers]
        elif hasattr(self.decoder_model.base_model.encoder, 'block'):
            decoder_stack = [self.decoder_model.base_model.encoder.block,
                             self.decoder_model.base_model.decoder.block]

        for encoder_parameter in decoder_stack:
            for name, param in encoder_parameter.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

        embshape = self.decoder_model.config.d_model
        bottleneck = int(embshape / 2)
        self.adapters = nn.ModuleList()
        [[self.adapters.append(nn.Sequential(nn.LayerNorm(embshape), nn.Linear(embshape, bottleneck), nn.ReLU(),
                                             nn.Linear(bottleneck, embshape))) for _ in model_decoder_layers] for
         model_decoder_layers in decoder_stack]
        self.adapters.to(self.device)
        for s_i, s in enumerate(decoder_stack):
            for l_i, l in enumerate(s):
                l.register_forward_hook(lambda m, i, o: (self.adapters[s_i * len(s) + l_i](o[0]), o[1:]))


class SpeechMixSelf(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()

        for name, param in self.decoder_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        if labels is not None:
            labels = labels.to(self.device)
        self.decoder_model.eval()
        outputs = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     decoder_input_ids=decoder_input_ids, labels=labels)
        if labels is not None:
            nlp_outputs = self.decoder_model(input_ids=text_input_ids, output_hidden_states=True,
                                             decoder_input_ids=decoder_input_ids, labels=labels)

            nlp_hidden = nlp_outputs['encoder_hidden_states'][-1]
            speech_hidden = outputs['encoder_hidden_states'][-1]
            attn_output = torch.bmm(nlp_hidden,
                                    speech_hidden.view(nlp_hidden.shape[0], self.decoder_model.config.hidden_size, -1))
            softmax = torch.nn.Softmax(dim=-1)
            attn_output = softmax(attn_output / math.sqrt(self.decoder_model.config.hidden_size))
            voice_projected_embeds = torch.bmm(attn_output, speech_hidden)
            mse_loss_fn = torch.nn.MSELoss()
            mse_loss = mse_loss_fn(voice_projected_embeds, nlp_hidden)

            kld_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            kld_loss = kld_loss_fn(torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                                   torch.nn.functional.softmax(nlp_outputs.logits, dim=-1))
            ce_loss = outputs.loss
            loss = kld_loss + ce_loss + mse_loss

            # print(mse_loss.mean().item(), kld_loss.mean().item(), ce_loss.mean().item())
            outputs['loss'] = loss.mean()

        return outputs


class SpeechMixGAN(SpeechMixEED):

    def custom_modules(self, **kwargs):
        self.encoder_model.eval()
        self.decoder_model.eval()

        for name, param in self.decoder_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

        self.discriminator = nn.Linear(self.decoder_model.config.hidden_size ** 2, 1).to(self.device)
        self.des_update = 1000
        self.update_count = 1
        self.keep_update = 1000

    def cal_loss(self, inputs_embeds=None, text_input_ids=None, attention_mask=None, decoder_input_ids=None,
                 labels=None):
        outputs = self.decoder_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                     output_hidden_states=True,
                                     decoder_input_ids=decoder_input_ids)
        loss = 0
        if labels is not None:
            if self.training:
                if self.update_count % self.des_update == 0:
                    if self.keep_update > 0:
                        self.keep_update -= 1
                        for name, p in self.named_parameters():
                            if 'discriminator' in name:
                                p.grad = None
                    else:
                        self.keep_update = 1000
                        self.update_count += 1
                else:
                    self.update_count += 1
                    for name, p in self.named_parameters():
                        if 'discriminator' not in name:
                            p.grad = None

            labels = labels.to(self.device)
            nlp_outputs = self.decoder_model(labels, output_hidden_states=True,
                                             decoder_input_ids=decoder_input_ids)

            voice_hidden = outputs['decoder_hidden_states'][-1]
            nlp_hidden = nlp_outputs['decoder_hidden_states'][-1]
            nlp_encoder_hidden = nlp_outputs['encoder_hidden_states'][-1]

            loss_fn = torch.nn.BCEWithLogitsLoss()
            voice_enc_attn_output = torch.bmm(
                inputs_embeds.view(inputs_embeds.shape[0], self.decoder_model.config.hidden_size, -1),
                inputs_embeds.view(inputs_embeds.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)
            vt_enc_loss = loss_fn(self.discriminator(voice_enc_attn_output).flatten(),
                                  torch.ones(voice_enc_attn_output.shape[0]).to(self.device))

            nlp_enc_attn_output = torch.bmm(
                nlp_encoder_hidden.view(nlp_encoder_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                nlp_encoder_hidden.view(nlp_encoder_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            nt_enc_loss = loss_fn(self.discriminator(nlp_enc_attn_output).flatten(),
                                  torch.zeros(nlp_enc_attn_output.shape[0]).to(self.device))

            voice_attn_output = torch.bmm(
                voice_hidden.view(voice_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                voice_hidden.view(voice_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            vt_loss = loss_fn(self.discriminator(voice_attn_output).flatten(),
                              torch.ones(voice_attn_output.shape[0]).to(self.device))

            nlp_attn_output = torch.bmm(
                nlp_hidden.view(nlp_hidden.shape[0], self.decoder_model.config.hidden_size, -1),
                nlp_hidden.view(nlp_hidden.shape[0], -1, self.decoder_model.config.hidden_size)) \
                .flatten(start_dim=1)

            nt_loss = loss_fn(self.discriminator(nlp_attn_output).flatten(),
                              torch.zeros(nlp_attn_output.shape[0]).to(self.device))

            loss += vt_loss + nt_loss + nt_enc_loss + vt_enc_loss
        outputs['loss'] = loss
        return outputs
