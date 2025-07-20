import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
import os
import sys
from typing import Dict, Any

# Ensure project root is in path to import model definitions
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.spirollm import SpiroLLM


def load_models_for_inference(spirollm_model_path: str, spiro_projection_path: str, torch_dtype: torch.dtype,
                              device: torch.device) -> tuple[SpiroLLM, Any]:
    """
    Loads all necessary models for inference on a single device.

    Args:
        merged_model_path (str): Path to the directory containing the merged LLM weights and tokenizer.
        spiro_projection_path (str): Path to the pre-trained spirometry projection layer weights.
        torch_dtype (torch.dtype): The data type for model loading (e.g., torch.bfloat16).
        device (torch.device): The device to load the models onto.

    Returns:
        tuple[SpiroLLM, Any]: A tuple containing the initialized SpiroLLM wrapper and the tokenizer.
    """
    logger.info(f"Loading all models onto a single device: {device}")

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer from {spirollm_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spirollm_model_path,
            trust_remote_code=True,
            use_fast=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.success("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Fatal error loading tokenizer: {e}")
        sys.exit(1)

    # 2. Load the base Large Language Model
    logger.info(f"Loading base LLM from {spirollm_model_path}...")
    llm = AutoModelForCausalLM.from_pretrained(
        spirollm_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        # Load on CPU first, then move to device to avoid potential `device_map` issues on single GPU
    ).to(device)
    logger.success("Base LLM loaded successfully.")

    # 3. Initialize SpiroLLM wrapper
    model_wrapper = SpiroLLM(
        base_model=llm,
        tokenizer=tokenizer,
        spiro_feature_dim=128,  # This should be read from config
        use_lora=False,  # LoRA weights are already merged
        is_pretrain_projector=False
    )

    # 4. Load projector weights and move the entire model to the target device
    model_wrapper.load_projector_weights(spiro_projection_path)
    model_wrapper.to(device)
    model_wrapper.eval()
    logger.success("SpiroLLM wrapper is fully loaded, configured, and set to evaluation mode.")

    return model_wrapper, tokenizer


def prepare_inputs_for_generation(batch: Dict, model: SpiroLLM, tokenizer: Any, device: torch.device,
                                  use_spiro: bool) -> Dict:
    """
    Prepares the final `inputs_embeds` and `attention_mask` for model generation.

    This function is called during inference to splice the spirometry feature embeddings
    into the text token embeddings.

    Args:
        batch (Dict): The batch data from the collate function.
        model (SpiroLLM): The SpiroLLM model instance.
        tokenizer (Any): The tokenizer instance.
        device (torch.device): The target device for tensors.
        use_spiro (bool): Flag indicating whether to use spirometry features.

    Returns:
        Dict: A dictionary with `inputs_embeds` and `attention_mask` ready for `model.generate()`.
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # If not using spiro features, we can pass input_ids directly, which is more efficient.
    if not use_spiro:
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # Get text embeddings
    token_embeds = model.base_model.get_input_embeddings()(input_ids)
    spiro_features = batch.get("spiro_features", None)
    has_spiro_features = batch.get("has_spiro_features", None)

    # If this batch has no spiro features, return text embeddings
    if spiro_features is None or not torch.any(has_spiro_features):
        return {"inputs_embeds": token_embeds, "attention_mask": attention_mask}

    # Project spirometry features to embedding space
    spiro_features = spiro_features.to(device)
    spiro_embeds = model.spiro_projection(spiro_features.to(token_embeds.dtype))

    special_token = model.get_special_feature_token()
    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
    special_token_indices = (input_ids == special_token_id).nonzero(as_tuple=False)

    final_embeds_list = []
    final_mask_list = []

    for i in range(input_ids.shape[0]):
        # Only perform splicing if the sample actually has features
        if has_spiro_features[i]:
            token_loc_row = (special_token_indices[:, 0] == i).nonzero()

            if token_loc_row.numel() > 0:
                insert_pos = special_token_indices[token_loc_row.item(), 1]
                spiro_to_insert = spiro_embeds[i]  # This will have seq_len of the feature
                spiro_mask_to_insert = torch.ones(spiro_to_insert.shape[0], dtype=torch.long, device=device)

                # Splice embeddings
                new_embed = torch.cat([token_embeds[i, :insert_pos], spiro_to_insert, token_embeds[i, insert_pos + 1:]],
                                      dim=0)
                # Splice attention mask
                new_mask = torch.cat(
                    [attention_mask[i, :insert_pos], spiro_mask_to_insert, attention_mask[i, insert_pos + 1:]], dim=0)

                final_embeds_list.append(new_embed)
                final_mask_list.append(new_mask)
            else:
                # Fallback: feature flag is true, but token not found
                final_embeds_list.append(token_embeds[i])
                final_mask_list.append(attention_mask[i])
        else:
            # No feature for this sample
            final_embeds_list.append(token_embeds[i])
            final_mask_list.append(attention_mask[i])

    # Pad the lists to form the final batch tensors
    final_inputs_embeds = pad_sequence(final_embeds_list, batch_first=True, padding_value=0.0)
    final_attention_mask = pad_sequence(final_mask_list, batch_first=True, padding_value=0)

    return {"inputs_embeds": final_inputs_embeds, "attention_mask": final_attention_mask}