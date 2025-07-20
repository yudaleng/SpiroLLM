import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, PeftModel, LoraConfig
from loguru import logger
import os


class SpiroLLM(nn.Module):
    """
    A wrapper for a large language models (LLM) that enhances it with the ability
    to process and integrate spirometry features.

    This class takes a pre-loaded base LLM and adds a projection layer to map
    spirometry features into the models's embedding space. These feature embeddings
    are then inserted into the token embedding sequence at a specified special
    token location.
    """

    def __init__(self,
                 base_model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 spiro_feature_dim: int,
                 use_lora: bool = True,
                 lora_config: LoraConfig = None,
                 is_pretrain_projector: bool = False):
        """
        Initializes the SpiroLLM wrapper.

        Args:
            base_model (PreTrainedModel): The pre-loaded base language models.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the base models.
            spiro_feature_dim (int): The dimensionality of the input spirometry features.
            use_lora (bool): Whether to apply LoRA for fine-tuning. Defaults to True.
            lora_config (LoraConfig, optional): Configuration for LoRA. Required if use_lora is True.
            is_pretrain_projector (bool): If True, freezes the base models and only trains the
                                          projection layer. Defaults to False.
        """
        super().__init__()

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.use_lora = use_lora

        logger.info("SpiroLLM wrapper initialized with a pre-loaded base models.")

        model_dtype = next(self.base_model.parameters()).dtype
        self.hidden_size = self.base_model.config.hidden_size

        # Define the projection layer for spirometry features
        self.spiro_projection = nn.Sequential(
            nn.Linear(spiro_feature_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        ).to(dtype=model_dtype)
        logger.info(f"Initialized Spiro Projection Layer with dtype: {model_dtype}")

        if is_pretrain_projector:
            logger.info("Projector pre-training mode: Freezing all base models weights.")
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.spiro_projection.parameters():
                param.requires_grad = True
        elif self.use_lora:
            if lora_config is None:
                raise ValueError("lora_config must be provided if use_lora is True.")
            self.base_model = get_peft_model(self.base_model, lora_config)
            logger.info("LoRA applied to the base models successfully.")
            self.base_model.print_trainable_parameters()

        self.input_embeddings = self.base_model.get_input_embeddings()

    def load_projector_weights(self, weights_path: str):
        """
        Loads pre-trained weights for the spirometry projection layer.

        Args:
            weights_path (str): Path to the pre-trained weights file.
        """
        if not os.path.exists(weights_path):
            logger.warning(
                f"Projector weights path not found: {weights_path}. Using random initialization."
            )
            return

        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Handle potential state_dict naming variations
            if 'spiro_projection.0.weight' in state_dict:
                projector_state = {k.replace('spiro_projection.', ''): v for k, v in state_dict.items() if
                                   k.startswith('spiro_projection.')}
            else:
                projector_state = state_dict

            self.spiro_projection.load_state_dict(projector_state, strict=False)
            model_dtype = next(self.base_model.parameters()).dtype
            self.spiro_projection.to(dtype=model_dtype)
            logger.info(f"Successfully loaded projector weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load projector weights from {weights_path}: {e}")

    def get_special_feature_token(self) -> str:
        """Dynamically gets the special token for feature insertion."""
        # Assumes the first additional special token is the one for features
        if self.tokenizer.additional_special_tokens:
            return self.tokenizer.additional_special_tokens[0]
        raise ValueError("No additional special tokens found in the tokenizer for feature insertion.")

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                spiro_features=None, has_spiro_features=None, **kwargs):

        if input_ids is None and kwargs.get('inputs_embeds') is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        # If no spirometry features are present, proceed with the standard base models forward pass.
        if spiro_features is None or not torch.any(has_spiro_features):
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

        # 1. Get token embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # 2. Project spirometry features
        spiro_embeds = self.spiro_projection(spiro_features.to(inputs_embeds.dtype))
        spiro_seq_len = spiro_embeds.shape[1]

        # 3. Find insertion points (location of the special token)
        special_token = self.get_special_feature_token()
        special_token_id = self.tokenizer.convert_tokens_to_ids(special_token)
        special_token_indices = (input_ids == special_token_id).nonzero(as_tuple=False)

        # Map each sample in the batch to its corresponding spirometry feature
        spiro_embeds_map = (torch.cumsum(has_spiro_features, dim=0) - 1).long()

        new_embeds_list, new_mask_list, new_labels_list = [], [], []

        for i in range(inputs_embeds.shape[0]):
            current_embed = inputs_embeds[i]
            current_mask = attention_mask[i]
            current_labels = labels[i] if labels is not None else None

            if has_spiro_features[i]:
                # Find the specific token location for this sample
                token_loc_row = (special_token_indices[:, 0] == i).nonzero()
                if token_loc_row.numel() > 0:
                    insert_pos = special_token_indices[token_loc_row.item(), 1]
                    spiro_idx = spiro_embeds_map[i]
                    spiro_block_to_insert = spiro_embeds[spiro_idx]

                    # 4. Splice the embeddings, attention mask, and labels
                    embed_before = current_embed[:insert_pos]
                    embed_after = current_embed[insert_pos + 1:]
                    new_embed = torch.cat([embed_before, spiro_block_to_insert, embed_after], dim=0)
                    new_embeds_list.append(new_embed)

                    mask_before = current_mask[:insert_pos]
                    mask_after = current_mask[insert_pos + 1:]
                    spiro_mask_to_insert = torch.ones(spiro_seq_len, dtype=torch.long, device=inputs_embeds.device)
                    new_mask = torch.cat([mask_before, spiro_mask_to_insert, mask_after], dim=0)
                    new_mask_list.append(new_mask)

                    if current_labels is not None:
                        labels_before = current_labels[:insert_pos]
                        labels_after = current_labels[insert_pos + 1:]
                        # Ignore loss for the inserted feature part by using -100
                        spiro_labels_to_insert = torch.full((spiro_seq_len,), -100, dtype=torch.long,
                                                            device=labels.device)
                        new_label = torch.cat([labels_before, spiro_labels_to_insert, labels_after], dim=0)
                        new_labels_list.append(new_label)
                else:
                    # Fallback if special token not found for a sample that should have it
                    logger.warning(
                        f"Sample {i} was marked to have spiro features, but the special token was not found.")
                    # Handle by just appending the original, un-padded data
                    new_embeds_list.append(current_embed)
                    new_mask_list.append(current_mask)
                    if current_labels is not None:
                        new_labels_list.append(current_labels)
            else:
                new_embeds_list.append(current_embed)
                new_mask_list.append(current_mask)
                if current_labels is not None:
                    new_labels_list.append(current_labels)

        # 5. Pad the lists to form new batch tensors
        final_inputs_embeds = pad_sequence(new_embeds_list, batch_first=True, padding_value=0.0)
        final_attention_mask = pad_sequence(new_mask_list, batch_first=True, padding_value=0)
        final_labels = pad_sequence(new_labels_list, batch_first=True,
                                    padding_value=-100) if new_labels_list and labels is not None else None

        return self.base_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            **kwargs
        )