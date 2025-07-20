import os
import sys
import requests
from loguru import logger


def fetch_ukbb_data(url: str) -> str:
    """
    Fetches the raw data as a string from the given URL.

    Args:
        url (str): The URL to fetch data from.

    Returns:
        str: The raw data as a string, or an empty string if fetching fails.
    """
    logger.info(f"Fetching data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        logger.success("Data fetched successfully.")
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data: {e}")
        sys.exit(1)


def save_data_to_csv(data: str, file_path: str):
    """
    Saves the given data string to a specified file path.

    Args:
        data (str): The data string to save.
        file_path (str): The path to save the file to, including the filename.
    """
    try:
        # Ensure the parent directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Write the data content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)
        logger.success(f"Data successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save file: {e}")
        sys.exit(1)


def main():
    """
    Main function to orchestrate fetching and saving the UK Biobank example data.
    """
    # URL for the example spirometry data from UK Biobank
    example_dataset_url = "https://biobank.ndph.ox.ac.uk/showcase/ukb/examples/eg_spiro_3066.dat"

    # Define the output path for the CSV file within a 'data' subdirectory
    output_csv_path = os.path.join("data", "example.csv")

    # Fetch the raw data from the URL
    ukbb_data = fetch_ukbb_data(example_dataset_url)

    # Save the fetched data to the specified CSV file
    save_data_to_csv(ukbb_data, output_csv_path)


if __name__ == "__main__":
    main()
