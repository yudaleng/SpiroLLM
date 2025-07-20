import numpy as np
from loguru import logger
from typing import Dict, Any, Optional
import json
from spiref import gli12


class SpiroProcessor:
    """
    A utility class for processing and analyzing spirometry (lung function test) data.
    It calculates key metrics, compares them against GLI-2012 reference values,
    and generates a comprehensive JSON report.
    """

    # --- Class Constants ---
    TIME_SCALE = 0.01  # Time step in seconds, corresponding to a 100Hz sampling rate.
    VOLUME_SCALE = 0.001  # Conversion factor from raw data points to Liters (L).
    ETHNICITY_MAP = {
        'caucasian': 'Cau',
        'africanamerican': 'AfrAm',
        'northeastasian': 'NEAsia',
        'southeastasian': 'SEAsia',
        'other': 'other'
    }

    def __init__(self):
        """Initializes the SpiroProcessor and the GLI-2012 calculator."""
        self.calculator = gli12.GLIReferenceValueCalculator()
        self._reset_state()

    def analyze(self,
                csv_path: str,
                sex: str,
                age: float,
                height_cm: float,
                is_smoker: bool,
                ethnicity: str = 'Caucasian',
                max_interp_volume: float = 8.0
                ) -> Optional[Dict[str, Any]]:
        """
        Main public method to perform a full analysis from a CSV file.

        Args:
            csv_path (str): The path to the input CSV file containing raw series data.
            sex (str): Subject's sex ('Male' or 'Female').
            age (float): Subject's age in years.
            height_cm (float): Subject's height in centimeters.
            is_smoker (bool): Smoking status of the subject.
            ethnicity (str): Subject's ethnicity. Defaults to 'Caucasian'.
            max_interp_volume (float): The maximum volume (L) for interpolation.

        Returns:
            A dictionary containing the full analysis results, or None if processing fails.
        """
        logger.info(f"--- Starting analysis ---")
        try:
            self._set_demographics(sex, age, height_cm, is_smoker, ethnicity)
            self._calculate_predictions_and_lln()

            with open(csv_path, 'r') as f:
                first_line = f.readline()
            if not first_line:
                logger.error(f"CSV file is empty: {csv_path}")
                return None

            raw_series = self.parse_series_from_string(first_line)
            if raw_series.size == 0:
                logger.error(f"Failed to parse valid series data from: {csv_path}")
                return None

            # --- Core Calculation Workflow ---
            self._calculate_measured_values(raw_series)
            self._calculate_derived_metrics()
            pft_results = self._format_pft_results()

            flow_volume_processed = self.compute_flow_volume_by_num_points(
                series=raw_series,
                volume_scale=self.VOLUME_SCALE,
                time_scale=self.TIME_SCALE,
                max_interp_volume=max_interp_volume
            )

            final_result = {
                "pft_json": {"BasicInfo": self.basic_info, "PFT_Results": pft_results},
                "flow_volume": [float(x) for x in flow_volume_processed]
            }

            logger.success(f"--- Successfully completed analysis ---")
            return final_result

        except Exception as e:
            logger.error(f"An error occurred while processing {csv_path}", exc_info=True)
            return None

    def _reset_state(self):
        """Resets all instance state variables to process a new file."""
        self.sex: Optional[str] = None
        self.age: Optional[float] = None
        self.height_cm: Optional[float] = None
        self.spiref_ethnicity: Optional[str] = None
        self.is_smoker: Optional[str] = None
        self.basic_info: Dict[str, Any] = {}
        self.measured: Dict[str, float] = {}
        self.predicted: Dict[str, float] = {}
        self.lln: Dict[str, float] = {}
        self.z_scores: Dict[str, float] = {}
        self.percent_predicted: Dict[str, float] = {}

    def _set_demographics(self, sex: str, age: float, height_cm: float, is_smoker: bool, ethnicity: str):
        """Validates and sets the demographic information for the subject."""
        self._reset_state()
        if sex.lower() not in ['male', 'female']:
            raise ValueError("Sex must be 'male' or 'female'.")

        ethnicity_lower = ethnicity.lower().replace(" ", "")
        if ethnicity_lower not in self.ETHNICITY_MAP:
            raise ValueError(
                f"Unsupported ethnicity: '{ethnicity}'. Supported options: {list(self.ETHNICITY_MAP.keys())}")

        self.sex = sex.lower()
        self.age = age
        self.height_cm = height_cm
        self.spiref_ethnicity = self.ETHNICITY_MAP[ethnicity_lower]
        self.is_smoker = "Yes" if is_smoker else "No"

        self.basic_info = {
            "Sex": self.sex.capitalize(),
            "Age": int(self.age),
            "Height_cm": float(self.height_cm),
            "IsSmoker": self.is_smoker
        }
        logger.info(
            f"Demographics set: Age={self.age}, Sex={self.sex}, Height={self.height_cm}cm, Ethnicity={ethnicity} ({self.spiref_ethnicity})")

    def _calculate_predictions_and_lln(self):
        """Calculates predicted values and Lower Limits of Normal (LLN) using spiref."""
        logger.info("Calculating Predicted values and LLN...")
        params_to_calc = {
            'FEV1': 'FEV1', 'FVC': 'FVC', 'FEV1_FVC': 'FEV1FVC', 'FEF25_75': 'FEF2575'
        }
        for key, spiref_key in params_to_calc.items():
            try:
                # Dynamically get the calculation methods from the calculator instance
                pred_func = getattr(self.calculator, f'calculate_{spiref_key.lower()}')

                pred_value = pred_func(self.sex, self.height_cm, self.age, race=self.spiref_ethnicity)
                lln_value = self.calculator.calc_lln_lung_param(spiref_key, self.sex, self.height_cm, self.age,
                                                                race=self.spiref_ethnicity)

                self.predicted[key] = pred_value
                self.lln[key] = lln_value
            except Exception as e:
                logger.warning(f"Could not calculate prediction/LLN for {key} with spiref: {e}")

    def _calculate_measured_values(self, raw_series: np.ndarray):
        """Calculates measured spirometry parameters from the raw data series."""
        logger.info("Calculating measured values from raw series...")
        volume_curve = raw_series * self.VOLUME_SCALE

        if len(volume_curve) < 10:
            logger.warning("Volume series is too short to be processed. Measured values will be empty.")
            self.measured = {}
            return

        sampling_rate = 1 / self.TIME_SCALE
        flow_curve = np.diff(volume_curve, prepend=0) * sampling_rate

        fvc = np.max(volume_curve)
        if fvc <= 0:
            logger.error("Calculated FVC is zero or negative. Cannot proceed with calculations.")
            self.measured = {}
            return

        # FEV1 is the volume exhaled in the first second
        fev1_index = min(int(1 * sampling_rate), len(volume_curve) - 1)
        fev1 = volume_curve[fev1_index]

        # PEF is the maximum flow rate in L/s, converted to L/min for output
        pef_l_per_sec = np.max(flow_curve)

        # Calculate FEF25-75
        fef25_75 = 0
        try:
            vol_25, vol_75 = fvc * 0.25, fvc * 0.75
            idx_25 = np.where(volume_curve >= vol_25)[0][0]
            idx_75 = np.where(volume_curve >= vol_75)[0][0]
            if idx_75 > idx_25:
                time_diff = (idx_75 - idx_25) * self.TIME_SCALE
                volume_diff = volume_curve[idx_75] - volume_curve[idx_25]
                fef25_75 = volume_diff / time_diff
        except IndexError:
            logger.warning("Could not determine FEF25-75, likely due to insufficient exhalation volume.")
            fef25_75 = 0

        self.measured = {
            'FVC': fvc,
            'FEV1': fev1,
            'PEF': pef_l_per_sec * 60,  # PEF is often reported in L/min
            'FEF25_75': fef25_75,
            'FEV1_FVC': fev1 / fvc if fvc > 0 else 0
        }

    def _calculate_derived_metrics(self):
        """Calculates Z-scores and Percent-of-Predicted values."""
        logger.info("Calculating Z-scores and Percent-of-Predicted...")
        for param, measured_val in self.measured.items():
            if param in self.predicted and self.predicted[param] > 0:
                pred, lln = self.predicted[param], self.lln[param]
                # Standard deviation (SD) is derived from the fact that LLN = Pred - 1.645 * SD
                sd = (pred - lln) / 1.645
                self.percent_predicted[param] = (measured_val / pred) * 100
                if sd > 0:
                    self.z_scores[param] = (measured_val - pred) / sd

    def _format_pft_results(self) -> Dict[str, Any]:
        """Formats the calculated results into the final PFT_Results dictionary."""
        logger.info("Formatting final PFT results dictionary...")
        pft_results = {}
        all_params = ['FEV1', 'FVC', 'FEV1_FVC', 'PEF', 'FEF25_75']

        for param in all_params:
            if param not in self.measured:
                continue

            result_dict = {}
            if param == 'FEV1_FVC':
                result_dict['ratio'] = float(round(self.measured.get(param, 0), 3))
            elif param == 'PEF':
                result_dict['measured_L_min'] = float(round(self.measured.get(param, 0), 1))
            elif param == 'FEF25_75':
                result_dict['measured_L_s'] = float(round(self.measured.get(param, 0), 2))
            else:
                result_dict['measured_L'] = float(round(self.measured.get(param, 0), 2))

            if param in self.predicted:
                if param == 'FEV1_FVC':
                    result_dict['LLN_ratio'] = float(round(self.lln.get(param, 0), 3))
                elif param == 'FEF25_75':
                    result_dict['predicted_L_s'] = float(round(self.predicted.get(param, 0), 3))
                    result_dict['LLN_L_s'] = float(round(self.lln.get(param, 0), 3))
                elif param != 'PEF':
                    result_dict['predicted_L'] = float(round(self.predicted.get(param, 0), 3))
                    result_dict['LLN_L'] = float(round(self.lln.get(param, 0), 3))

                if param in self.z_scores:
                    result_dict['zscore'] = float(round(self.z_scores[param], 3))
                if param in self.percent_predicted:
                    result_dict['predicted_percent'] = float(round(self.percent_predicted[param], 3))

            if result_dict:
                pft_results[param] = result_dict

        return pft_results

    # --------------------------------------------------------------------------
    # --- Static Methods ---
    # --------------------------------------------------------------------------

    @staticmethod
    def parse_series_from_string(series_string: str) -> np.ndarray:
        """
        Parses a comma-separated string to extract numeric series data.
        Skips the first 3 values, which are typically metadata.
        """
        if not isinstance(series_string, str):
            raise TypeError("Input must be a string.")

        series_values = series_string.strip().split(',')[5:]
        valid_numeric_values = [int(v) for v in series_values if v.strip().lstrip('-').isdigit()]
        return np.array(valid_numeric_values, dtype=np.float32)

    @staticmethod
    def compute_flow_volume_by_num_points(series: np.ndarray,
                                          volume_scale: float,
                                          time_scale: float,
                                          max_interp_volume: float) -> np.ndarray:
        """
        Computes, interpolates, and pads a flow-volume curve from a raw series.

        Args:
            series (np.ndarray): The input raw data series.
            volume_scale (float): Factor to convert series data to volume in Liters.
            time_scale (float): Time step in seconds for flow calculation.
            max_interp_volume (float): The maximum volume (L) to interpolate up to.

        Returns:
            np.ndarray: The processed flow-volume curve array.
        """
        # Scale series to a volume curve in Liters (L)
        volume = (series * volume_scale).astype(np.float32)
        # Calculate flow from the volume curve
        flow = np.concatenate(([0.0], np.diff(volume) / time_scale))

        def right_pad_array(arr, pad_value, num_points):
            """Helper function to pad an array on the right."""
            if len(arr) >= num_points:
                return arr[:num_points]
            return np.pad(arr, (0, num_points - len(arr)), 'constant', constant_values=(pad_value,))

        # Pad original volume and flow arrays to ensure consistent length
        padded_volume = right_pad_array(volume, 0, len(volume))
        padded_flow = right_pad_array(flow, 0, len(flow))

        # Create the flow-volume curve by interpolation
        # Ensure volume is monotonic for correct interpolation
        monotonic_volume = np.maximum.accumulate(padded_volume)
        volume_interp_intervals = np.linspace(start=0, stop=max_interp_volume, num=len(flow))

        # Interpolate flow against the monotonic volume
        flow_volume_curve = np.interp(volume_interp_intervals, xp=monotonic_volume, fp=padded_flow, left=0, right=0)
        return flow_volume_curve


# ==============================================================================
# --- Example Usage ---
# ==============================================================================

if __name__ == '__main__':
    dummy_csv_path = "/data/meishuhao/SpiroLLM/data/example.csv"
    try:

        # Instantiate the processor
        processor = SpiroProcessor()

        # Call the main analysis function
        final_json_output = processor.analyze(
            csv_path=dummy_csv_path,
            sex='Male',
            age=69,
            height_cm=176.0,
            is_smoker=False,
            ethnicity='Caucasian',
            max_interp_volume=8.0
        )

        if final_json_output:
            print("\n--- ✅ Final JSON Generation Successful ---")

            # For brevity, print only a portion of the flow-volume data
            if 'flow_volume' in final_json_output:
                num_points = len(final_json_output['flow_volume'])
                print(f"Flow-Volume array contains {num_points} points.")
                # Create a preview of the array
                final_json_output['flow_volume'] = final_json_output['flow_volume'][:10] + ['...']

            print(json.dumps(final_json_output, indent=4))
        else:
            print("\n--- ❌ Final JSON Generation Failed ---")

    except FileNotFoundError:
        logger.error(f"Example file not found: {dummy_csv_path}. Please create it with sample data.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}")