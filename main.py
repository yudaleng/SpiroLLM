import os
import sys
import argparse
import json
import yaml
from huggingface_hub import snapshot_download
from loguru import logger
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from functools import partial
from typing import Dict, Any

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_utils import load_models_for_inference, prepare_inputs_for_generation
from utils.data_processor import get_spiro_features, inference_collate_fn
from utils.spiroUtils import SpiroProcessor
from datasets import Dataset


def run_inference(config: Dict[str, Any], test_dataset: Dataset):
    """
    Runs the complete inference pipeline, including model loading, data processing,
    and report generation.

    Args:
        config (Dict[str, Any]): Configuration loaded from the YAML file.
        test_dataset (Dataset): The Hugging Face Dataset to run inference on.
    """
    logger.info("--- Starting Inference Pipeline ---")

    # --- 1. Setup Device and Data Type ---
    device = torch.device(config['inference_params']['device'] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config['model_params']['torch_dtype'] == 'bf16' else torch.float16
    logger.info(f"Using device: {device} with dtype: {dtype}")

    # --- 2. Load Models and Tokenizer ---
    model, tokenizer = load_models_for_inference(
        spirollm_model_path=config['paths']['spirollm_model_path'],
        spiro_projection_path=config['paths']['spiro_projection_path'],
        torch_dtype=dtype,
        device=device
    )

    # --- 3. Pre-calculate Spirometry Features ---
    eid_to_spiro_features = {}
    if config['inference_params']['use_spiro_features']:
        logger.info("Pre-calculating spirometry features and COPD probabilities...")
        raw_data_list = [
            {"eid": item[config['data_fields']['eid']], "flow_volume_data": item[config['data_fields']['raw_signal']]}
            for item in test_dataset if item.get(config['data_fields']['raw_signal'])
        ]
        if raw_data_list:
            eid_to_spiro_features = get_spiro_features(
                raw_data_list=raw_data_list,
                model_weight_path=config['paths']['deepspiro_model_path'],
                config=config
            )
            logger.info(f"Finished pre-calculating features for {len(eid_to_spiro_features)} EIDs.")

    # --- 4. Create DataLoader ---
    collate_wrapper = partial(
        inference_collate_fn,
        tokenizer=tokenizer,
        use_spiro=config['inference_params']['use_spiro_features'],
        spiro_features_map=eid_to_spiro_features,
        config=config
    )
    dataloader = DataLoader(
        test_dataset,
        batch_size=config['inference_params']['batch_size'],
        collate_fn=collate_wrapper,
        num_workers=0
    )

    # --- 5. Run Inference and Save Results ---
    output_dir = os.path.dirname(config['paths']['output_file'])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []
    logger.info(f"Results will be saved to: {config['paths']['output_file']}")
    with open(config['paths']['output_file'], 'w', encoding='utf-8') as f_out:
        for batch in tqdm(dataloader, desc="Generating Reports"):
            generation_inputs = prepare_inputs_for_generation(
                batch=batch,
                model=model,
                tokenizer=tokenizer,
                device=device,
                use_spiro=config['inference_params']['use_spiro_features']
            )

            with torch.no_grad():
                generation_output = model.base_model.generate(
                    **generation_inputs,
                    **config['generation_params']
                )

            decoded_outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

            # Extract the assistant's response
            final_texts = []
            for i, full_text in enumerate(decoded_outputs):
                search_marker = "assistant\n"
                last_marker_pos = full_text.rfind(search_marker)
                if last_marker_pos != -1:
                    pred_text = full_text[last_marker_pos + len(search_marker):].strip()
                    final_texts.append(pred_text)
                else:
                    final_texts.append(full_text.strip())

            for i, eid in enumerate(batch["eid"]):
                result = {"eid": eid, "generated_report": final_texts[i]}
                all_results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.success("Inference complete!")
    return all_results


def main():
    """
    Main entry point: parses arguments and starts the inference for a single patient.
    """
    parser = argparse.ArgumentParser(description="SpiroLLM: Generate a diagnostic report for a single patient.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Path to the inference configuration YAML file."
    )

    # --- Single Patient Input Arguments ---
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the patient's raw data CSV file.")
    parser.add_argument('--age', type=int, required=True, help="Patient's age in years.")
    parser.add_argument('--sex', type=str, choices=['Male', 'Female'], required=True,
                        help="Patient's sex ('Male' or 'Female').")
    parser.add_argument('--height_cm', type=float, required=True, help="Patient's height in centimeters.")
    parser.add_argument('--is_smoker', action='store_true', help="Flag if the patient is a smoker.")
    parser.add_argument('--ethnicity', type=str, default='Caucasian',
                        help="Patient's ethnicity (e.g., Caucasian, AfricanAmerican, NEAsia, SEAsia, Other).")

    args = parser.parse_args()

    # --- 1. Load Configuration ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        sys.exit(1)

    local_model_path = config['paths']['spirollm_model_path']
    repo_id = config['paths'].get('spirollm_repo_id')

    # Check if the local directory seems valid (e.g., contains config.json)
    model_config_path = os.path.join(local_model_path, 'config.json')

    if not os.path.exists(model_config_path):
        logger.info(f"Local model not found at '{local_model_path}'.")
        if repo_id:
            logger.info(f"Downloading from Hugging Face Hub: '{repo_id}'...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False
                )
                logger.success(f"Model downloaded successfully to '{local_model_path}'.")
            except Exception as e:
                logger.error(f"Failed to download model from Hugging Face Hub: {e}")
                sys.exit(1)
        else:
            logger.error(
                f"Local model not found and 'spirollm_repo_id' is not specified in the config. Cannot proceed.")
            sys.exit(1)
    else:
        logger.info(f"Found existing local model at '{local_model_path}'. Skipping download.")

    # Update config to use the verified local path for the rest of the script
    config['paths']['spirollm_model_path'] = local_model_path

    # --- 2. Process Single Patient Data ---
    logger.info("--- Single Patient Input Mode ---")

    # Process single patient data using SpiroProcessor
    processor = SpiroProcessor()
    analysis_result = processor.analyze(
        csv_path=args.csv_path,
        sex=args.sex,
        age=args.age,
        height_cm=args.height_cm,
        is_smoker=args.is_smoker,
        ethnicity=args.ethnicity
    )

    if not analysis_result:
        logger.error("Data processing failed. Please check the CSV file and input parameters.")
        sys.exit(1)

    single_patient_data = {
        config['data_fields']['eid']: ["demo_patient"],
        config['data_fields']['pft_text']: [analysis_result["pft_json"]],
        config['data_fields']['raw_signal']: [analysis_result["flow_volume"]]
    }
    test_dataset = Dataset.from_dict(single_patient_data)
    logger.info("Single patient data processed. Preparing for model input.")

    # --- 3. Run Inference ---
    results = run_inference(config, test_dataset)

    # --- 4. Print Final Report ---
    if results:
        logger.info("--- Generated Diagnostic Report ---")
        print(json.dumps(results[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
