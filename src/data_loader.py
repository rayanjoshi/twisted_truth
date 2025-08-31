import os
from pathlib import Path
from typing import Optional
import logging
from omegaconf import DictConfig
import hydra
from dotenv import load_dotenv
from fredapi import Fred
import yfinance as yf
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger("data_loader")

class LoadData:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.series_id = cfg.load_data.fred_series_id
        self.start_date = cfg.load_data.start_date
        self.end_date = cfg.load_data.end_date
        self.save_path = cfg.load_data.output_file
        self.index_ticker = cfg.load_data.index_ticker

        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        self.save_path = Path(repo_root / self.save_path).resolve()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

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
            logger.info("Fetching M1 money supply data from FRED...")
            m1_data = fred.get_series(self.series_id,
                                        observation_start=self.start_date,
                                        observation_end=self.end_date
                                        )
            logger.info("Retrieved %d records from FRED.", len(m1_data))
            m1_df = pd.DataFrame(m1_data, columns=['M1_Money_Supply'])
            logger.debug("DataFrame columns: %s", m1_df.columns)
            m1_df.index.name = 'Date'
            logger.debug("DataFrame head:\n%s", m1_df.head())

            m1_df.to_csv(self.save_path, index=True)
            logger.info("M1 money supply data saved to %s.", self.save_path)

        except (KeyError, ValueError, OSError) as e:
            logger.error("An error occurred: %s", e)
            raise

    def get_index_data(self):
        """
        Downloads index data, combines it with existing M1 data, and saves the result.

        This method retrieves historical index data from Yahoo Finance for the specified
        ticker and date range, aligns it with existing M1 data from a CSV file, handles
        missing values, and saves the combined dataset back to the CSV file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the CSV file at `self.save_path` does not exist.
            pandas.errors.EmptyDataError: If the CSV file is empty or corrupt.

        Notes:
            - The index data is retrieved with `auto_adjust=True` to adjust for splits/dividends.
            - Missing values in the combined dataset are dropped before saving.
        """
        try:
            data = yf.download(self.index_ticker,
                                start=self.start_date,
                                end=self.end_date,
                                multi_level_index=False,
                                auto_adjust=True,
                                )
            logger.info("Retrieved %d records from Yahoo Finance for index: %s.",
                        len(data), self.index_ticker)

            index_df = data['Close'].rename('Index_Close')
            index_df.index = pd.to_datetime(index_df.index)
            index_df.index.name = 'Date'

            logger.info("Loading existing M1 data from %s...", self.save_path)
            m1_df = pd.read_csv(self.save_path, index_col='Date', parse_dates=True)

            logger.info("Aligning and concatenating M1 data with index data...")
            combined_df = pd.concat([m1_df, index_df], axis=1, join='outer')
            logger.debug("Combined DataFrame head:\n%s", combined_df.head())
            nan_count = combined_df.isna().sum().sum()

            if nan_count > 0:
                logger.warning("Found %d NaN values.", nan_count)
            combined_df.dropna(inplace=True)
            logger.debug("Rows remaining after NaN removal: %d", combined_df.shape[0])

            combined_df.to_csv(self.save_path, index=True)
            logger.info("Appended index data to %s.", self.save_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error("An error occurred while processing the data: %s", e)
            raise


@hydra.main(version_base=None, config_path="../configs", config_name="data_loader")
def main(cfg: Optional[DictConfig]=None):
    data_loader = LoadData(cfg)
    data_loader.get_m1_data()
    data_loader.get_index_data()

if __name__ == "__main__":
    main()
