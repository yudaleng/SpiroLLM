# --- Constants for Prompt Structure ---
SYSTEM_PROMPT = "You are an expert in COPD diagnosis."

USER_PROMPT_TEMPLATE = """Spiro multimodal feature embeddings: 
<DeepSpiro_Feature> 
Spiro_PFT_Data (JSON): 
{pft_json_data} 
DeepSpiro predicted COPD probability: {copd_probability} 

Based on all the provided medical information (the embedded Spiro features and the JSON PFT data), please generate a comprehensive COPD diagnosis report.
"""

# Using the Llama 3 prompt format
PROMPT_STRUCTURE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Special tokens and placeholders
SPIRO_FEATURE_TOKEN = "<DeepSpiro_Feature>"

def format_diagnosis_prompt(pft_json: str, copd_prob: float) -> str:
    """
    Formats the complete prompt for the COPD diagnosis task.

    Args:
        pft_json (str): A string containing the PFT data in JSON format.
        copd_prob (float): The COPD probability predicted by the DeepSpiro models.

    Returns:
        str: A fully formatted prompt string ready for the LLM.
    """
    # 1. Create the user-facing query by filling in the data
    user_query = USER_PROMPT_TEMPLATE.format(
        pft_json_data=pft_json,
        copd_probability=f"{copd_prob:.4f}"
    )

    # 2. Construct the final prompt using the overall structure
    final_prompt = PROMPT_STRUCTURE.format(
        system_message=SYSTEM_PROMPT,
        user_query=user_query
    )

    return final_prompt