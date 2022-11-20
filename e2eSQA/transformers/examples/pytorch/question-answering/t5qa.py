# +
from transformers import T5PreTrainedModel, BertPreTrainedModel,PreTrainedModel, T5Model, LongformerModel
from typing import List, Optional, Tuple, Union
import torch
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from torch import nn

# NOTE : when inheriting from T5PretrainedModel, the code is met with : model parallism = False error
# inheriting from BERT as a temporary fix 
class T5ForQuestionAnswering(BertPreTrainedModel):
	def __init__(self, config, path, real_config):
		super().__init__(config)
		self.num_labels = real_config.num_labels
		#print(self.num_labels)
		#s()
		self.bert = None
		self.qa_outputs = nn.Linear(real_config.hidden_size, real_config.num_labels)

		# Initialize weights and apply final processing
		self.post_init()
        
		# Initialize T5 last as T5 initialization is not supported in post_init
		print(path)
		
		#self.encoder = LongformerModel.from_pretrained(path)#T5Model.from_pretrained(path).encoder
		#s()

	def forward(
		self,
		input_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		#token_type_ids: Optional[torch.Tensor] = None,
		#position_ids: Optional[torch.Tensor] = None,
		head_mask: Optional[torch.Tensor] = None,
		inputs_embeds: Optional[torch.Tensor] = None,
		start_positions: Optional[torch.Tensor] = None,
		end_positions: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
	) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
		r"""
		start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for position (index) of the start of the labelled span for computing the token classification loss.
			Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
			are not taken into account for computing the loss.
		end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
			Labels for position (index) of the end of the labelled span for computing the token classification loss.
			Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
			are not taken into account for computing the loss.
		"""
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# outputs = self.T5Model(
		# 	input_ids,
		# 	attention_mask=attention_mask,
		# 	token_type_ids=token_type_ids,
		# 	position_ids=position_ids,
		# 	head_mask=head_mask,
		# 	inputs_embeds=inputs_embeds,
		# 	output_attentions=output_attentions,
		# 	output_hidden_states=output_hidden_states,
		# 	return_dict=return_dict,
		# )
		outputs = self.encoder(
	        input_ids=input_ids,
	        attention_mask=attention_mask,
	        inputs_embeds=inputs_embeds,
	        head_mask=head_mask,
	        output_attentions=output_attentions,
	        output_hidden_states=output_hidden_states,
	        return_dict=return_dict,
	    )


		sequence_output = outputs[0]
		#print(sequence_output)
		#print(sequence_output.shape)
		#s()

		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1).contiguous()
		end_logits = end_logits.squeeze(-1).contiguous()

		total_loss = None
		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension
			if len(start_positions.size()) > 1:
				start_positions = start_positions.squeeze(-1)
			if len(end_positions.size()) > 1:
				end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions = start_positions.clamp(0, ignored_index)
			end_positions = end_positions.clamp(0, ignored_index)

			loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2

		if not return_dict:
			output = (start_logits, end_logits) + outputs[2:]
			return ((total_loss,) + output) if total_loss is not None else output

		return QuestionAnsweringModelOutput(
			loss=total_loss,
			start_logits=start_logits,
			end_logits=end_logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
# -



# class BertForQuestionAnswering(BertPreTrainedModel):

# 	_keys_to_ignore_on_load_unexpected = [r"pooler"]

# 	def __init__(self, config):
# 		super().__init__(config)
# 		self.num_labels = config.num_labels

# 		self.bert = BertModel(config, add_pooling_layer=False)
# 		self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

# 		# Initialize weights and apply final processing
# 		self.post_init()

# 	@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
# 	@add_code_sample_docstrings(
# 		processor_class=_TOKENIZER_FOR_DOC,
# 		checkpoint=_CHECKPOINT_FOR_QA,
# 		output_type=QuestionAnsweringModelOutput,
# 		config_class=_CONFIG_FOR_DOC,
# 		qa_target_start_index=_QA_TARGET_START_INDEX,
# 		qa_target_end_index=_QA_TARGET_END_INDEX,
# 		expected_output=_QA_EXPECTED_OUTPUT,
# 		expected_loss=_QA_EXPECTED_LOSS,
# 	)
# 	def forward(
# 		self,
# 		input_ids: Optional[torch.Tensor] = None,
# 		attention_mask: Optional[torch.Tensor] = None,
# 		token_type_ids: Optional[torch.Tensor] = None,
# 		position_ids: Optional[torch.Tensor] = None,
# 		head_mask: Optional[torch.Tensor] = None,
# 		inputs_embeds: Optional[torch.Tensor] = None,
# 		start_positions: Optional[torch.Tensor] = None,
# 		end_positions: Optional[torch.Tensor] = None,
# 		output_attentions: Optional[bool] = None,
# 		output_hidden_states: Optional[bool] = None,
# 		return_dict: Optional[bool] = None,
# 	) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
# 		r"""
# 		start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
# 			Labels for position (index) of the start of the labelled span for computing the token classification loss.
# 			Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
# 			are not taken into account for computing the loss.
# 		end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
# 			Labels for position (index) of the end of the labelled span for computing the token classification loss.
# 			Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
# 			are not taken into account for computing the loss.
# 		"""
# 		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

# 		outputs = self.bert(
# 			input_ids,
# 			attention_mask=attention_mask,
# 			token_type_ids=token_type_ids,
# 			position_ids=position_ids,
# 			head_mask=head_mask,
# 			inputs_embeds=inputs_embeds,
# 			output_attentions=output_attentions,
# 			output_hidden_states=output_hidden_states,
# 			return_dict=return_dict,
# 		)

# 		sequence_output = outputs[0]

# 		logits = self.qa_outputs(sequence_output)
# 		start_logits, end_logits = logits.split(1, dim=-1)
# 		start_logits = start_logits.squeeze(-1).contiguous()
# 		end_logits = end_logits.squeeze(-1).contiguous()

# 		total_loss = None
# 		if start_positions is not None and end_positions is not None:
# 			# If we are on multi-GPU, split add a dimension
# 			if len(start_positions.size()) > 1:
# 				start_positions = start_positions.squeeze(-1)
# 			if len(end_positions.size()) > 1:
# 				end_positions = end_positions.squeeze(-1)
# 			# sometimes the start/end positions are outside our model inputs, we ignore these terms
# 			ignored_index = start_logits.size(1)
# 			start_positions = start_positions.clamp(0, ignored_index)
# 			end_positions = end_positions.clamp(0, ignored_index)

# 			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
# 			start_loss = loss_fct(start_logits, start_positions)
# 			end_loss = loss_fct(end_logits, end_positions)
# 			total_loss = (start_loss + end_loss) / 2

# 		if not return_dict:
# 			output = (start_logits, end_logits) + outputs[2:]
# 			return ((total_loss,) + output) if total_loss is not None else output

# 		return QuestionAnsweringModelOutput(
# 			loss=total_loss,
# 			start_logits=start_logits,
# 			end_logits=end_logits,
# 			hidden_states=outputs.hidden_states,
# 			attentions=outputs.attentions,
# 		)

