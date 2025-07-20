import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from loguru import logger
import json
from typing import List, Dict, Any

# It's better to import specific modules/classes
from models.deepspiro import DeepSpiro
from prompts.copd_diagnosis_prompt import format_diagnosis_prompt


def get_spiro_features(raw_data_list: List[Dict[str, Any]], model_weight_path: str, config: Dict[str, Any]) -> Dict[
    str, Dict[str, Any]]:
    """
    Extracts high-dimensional features from raw spirometry data using the DeepSpiro model.

    Args:
        raw_data_list (List[Dict[str, Any]]): A list of dictionaries, each containing 'eid' and 'flow_volume_data'.
        model_weight_path (str): Path to the pre-trained DeepSpiro model weights.
        config (Dict[str, Any]): Configuration dictionary containing all necessary parameters.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping each 'eid' to its extracted features,
                                   segment mask, and predicted COPD probability.
    """
    device = torch.device(config['inference_params']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for spirometry feature extraction.")

    params = config['feature_extraction_params']
    processing_dtype = torch.float32
    output_dtype = torch.bfloat16 if config['model_params']['torch_dtype'] == 'bf16' else torch.float16

    model = DeepSpiro(
        in_channels=params['raw_signal_dim'],
        n_len_seg=params['n_len_seg'],
        n_classes=params['n_classes_model'],
        verbose=False  # Set to True for debugging shapes
    )
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    model.to(device=device, dtype=processing_dtype)
    model.eval()
    logger.info(f"DeepSpiro model loaded from {model_weight_path} and set to evaluation mode.")

    eid_to_features = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(raw_data_list), params['batch_size_deepspiro']),
                      desc="DeepSpiro Feature Extraction"):
            batch_raw_data = raw_data_list[i:i + params['batch_size_deepspiro']]

            eids_in_batch = [item['eid'] for item in batch_raw_data]
            # Ensure data is tensor and has 2 dimensions (len, features=1)
            flow_volume_tensors = []
            for item in batch_raw_data:
                tensor_data = torch.tensor(item['flow_volume_data'], dtype=processing_dtype)
                if tensor_data.ndim == 1:
                    tensor_data = tensor_data.unsqueeze(-1)  # Add feature/channel dimension
                flow_volume_tensors.append(tensor_data)

            padded_flow_volume = pad_sequence(flow_volume_tensors, batch_first=True, padding_value=0.0)
            actual_lengths = torch.tensor([t.shape[0] for t in flow_volume_tensors], dtype=torch.long)

            x_segmented, mask_for_crnn = _segment_physio_data(
                padded_flow_volume.to(device),
                actual_lengths.to(device),
                params['n_len_seg'],
                params['raw_signal_dim'],
                device,
                processing_dtype
            )

            features, logits = model(x_segmented, mask_for_crnn)
            probabilities = F.softmax(logits, dim=1)

            for j, eid in enumerate(eids_in_batch):
                num_valid_segments = int(mask_for_crnn[j].sum().item())
                if num_valid_segments > 0:
                    sample_features = features[j, :num_valid_segments, :].to(dtype=output_dtype).cpu()
                    eid_to_features[eid] = {
                        "features": sample_features,
                        "copd_probability": probabilities[j, 1].item()
                    }
                else:
                    logger.warning(f"EID {eid}: No valid segments found after processing.")
                    eid_to_features[eid] = {
                        "features": torch.empty(0, config['model_params']['spiro_feature_dim'], dtype=output_dtype),
                        "copd_probability": 0.0
                    }

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return eid_to_features


def _segment_physio_data(flow_volume_data: torch.Tensor, actual_lengths: torch.Tensor, n_len_seg: int,
                         raw_signal_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[
    torch.Tensor, torch.Tensor]:
    """
    Segments continuous physiological data into fixed-length windows.

    Args:
        flow_volume_data (torch.Tensor): Padded tensor of shape (batch, max_len, dim).
        actual_lengths (torch.Tensor): Tensor containing the actual length of each sample.
        n_len_seg (int): The length of each segment.
        raw_signal_dim (int): The dimension of the raw signal.
        device (torch.device): The device for computation.
        dtype (torch.dtype): The data type for tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Segmented data tensor and the corresponding mask.
    """
    batch_size = flow_volume_data.shape[0]
    all_segments, all_masks = [], []

    for i in range(batch_size):
        # ✨ --- FIX 1: Corrected indexing from 3D to 2D --- ✨
        # We only need two indices for a (batch, length, channels) tensor.
        # The third dimension (channels) is preserved by this slicing.
        current_len = actual_lengths[i]
        sample_data = flow_volume_data[i, :current_len]

        segments_for_sample = []
        if current_len > 0:
            # Use unfold to create non-overlapping segments
            # The shape will be (num_segments, raw_signal_dim, n_len_seg)
            segments_for_sample = sample_data.unfold(0, n_len_seg, n_len_seg).permute(0, 2, 1)

            # Handle the last partial segment manually
            last_seg_len = current_len % n_len_seg
            if last_seg_len > 0:
                last_segment = sample_data[-last_seg_len:]
                padding = torch.zeros((n_len_seg - last_seg_len, raw_signal_dim), device=device, dtype=dtype)
                padded_last_segment = torch.cat([last_segment, padding], dim=0)
                # Add the last segment as a new tensor in the list
                segments_for_sample = torch.cat([segments_for_sample, padded_last_segment.unsqueeze(0)], dim=0)

        if segments_for_sample.numel() == 0:  # Handle empty or very short samples
            all_masks.append(torch.tensor([0.0], device=device, dtype=torch.float))
            # Append a zero tensor with the correct final shape
            all_segments.append(torch.zeros((1, n_len_seg, raw_signal_dim), device=device, dtype=dtype))
        else:
            all_masks.append(torch.ones(segments_for_sample.shape[0], device=device, dtype=torch.float))
            all_segments.append(segments_for_sample)

    padded_segments = pad_sequence(all_segments, batch_first=True, padding_value=0.0)
    padded_masks = pad_sequence(all_masks, batch_first=True, padding_value=0.0)

    # The final shape for DeepSpiro is (batch, num_segments, n_len_seg, raw_signal_dim)
    return padded_segments, padded_masks


# The rest of the file (inference_collate_fn) remains the same.
def inference_collate_fn(batch: List[Dict], tokenizer: Any, use_spiro: bool, spiro_features_map: Dict,
                         config: Dict) -> Dict:
    pft_jsons = [json.dumps(item[config['data_fields']['pft_text']], ensure_ascii=False) for item in batch]
    eids = [item[config['data_fields']['eid']] for item in batch]
    prompts = []
    for i, pft_json_str in enumerate(pft_jsons):
        eid = eids[i]
        copd_prob = spiro_features_map.get(eid, {}).get("copd_probability", 0.0)
        prompt = format_diagnosis_prompt(pft_json_str, copd_prob)
        prompts.append(prompt)
    tokenized = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    batch_data = {"eid": eids, "input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}
    if use_spiro:
        spiro_features_list = []
        has_spiro_features_mask = []
        for eid in eids:
            spiro_data = spiro_features_map.get(eid)
            if spiro_data and spiro_data.get("features") is not None and spiro_data["features"].numel() > 0:
                spiro_features_list.append(spiro_data["features"])
                has_spiro_features_mask.append(True)
            else:
                spiro_features_list.append(torch.zeros(1, config['model_params']['spiro_feature_dim']))
                has_spiro_features_mask.append(False)
        batch_data["spiro_features"] = pad_sequence(spiro_features_list, batch_first=True, padding_value=0.0)
        batch_data["has_spiro_features"] = torch.tensor(has_spiro_features_mask, dtype=torch.bool)
    return batch_data