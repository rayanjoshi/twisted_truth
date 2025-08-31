import os
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

load_dotenv()

class LoadData:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.series_id = cfg.load_data.fred_series_id
        self.start_date = cfg.load_data.start_date
        self.end_date = cfg.load_data.end_date
        self.save_path = cfg.load_data.output_file

    def get_m1_data(self):
        """
        Retrieve M1 money supply data from FRED and save it to a CSV file.

        This method fetches M1 money supply data for a specified date range
        using the FRED API and saves the data as a CSV file to the designated
        save path.

        Attributes:
            series_id (str): The FRED series ID for M1 money supply data.
            start_date (str): The start date for data retrieval in 'YYYY-MM-DD' format.
            end_date (str): The end date for data retrieval in 'YYYY-MM-DD' format.
            save_path (str): The file path where the CSV file will be saved.

        Raises:
            KeyError: If the FRED_API_KEY environment variable is not set.
            ValueError: If the series_id, start_date, or end_date is invalid.
            OSError: If there are issues creating directories or writing the CSV file.

        Returns:
            None
        """
        try:
            fred_api_key = os.getenv("FRED_API_KEY")
            fred = Fred(api_key=fred_api_key)
            m1_data = fred.get_series(self.series_id,
                                        observation_start=self.start_date,
                                        observation_end=self.end_date
                                        )
            m1_df = pd.DataFrame(m1_data, columns=['M1_Money_Supply'])
            m1_df.index.name = 'Date'

            script_dir = Path(__file__).parent
            repo_root = script_dir.parent
            save_path = Path(repo_root / self.save_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            m1_df.to_csv(save_path, index=True)

        except (KeyError, ValueError, OSError) as e:
            print(f"An error occurred: {e}")
            raise


@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: Optional[DictConfig]=None):
    data_loader = LoadData(cfg)
    data_loader.get_m1_data()

if __name__ == "__main__":
    main()
