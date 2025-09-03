"""
Module for loading and processing financial data from FRED and Yahoo Finance.

This module provides functionality to retrieve M1 money supply data from the FRED API,
index data, and constituent stock data, and cryptocurrency data from Yahoo Finance. 
It combines these datasets into a single CSV file, handling data alignment and missing values. 
The module uses Hydra for configuration management and supports logging for debugging and 
monitoring.

The main entry point is the `main` function, which orchestrates the data loading process
using the `LoadData` class.

Dependencies:
    - os: For environment variable access.
    - pathlib: For file path manipulation.
    - typing: For type hints.
    - logging vlogging: For logging functionality.
    - omegaconf: For configuration management.
    - dotenv: For loading environment variables.
    - fredapi: For accessing the FRED API.
    - yfinance: For accessing Yahoo Finance data.
    - pandas: For data manipulation and storage.
"""
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader")

class LoadData:
    """A class to load and process financial data from FRED and Yahoo Finance.

    This class handles the retrieval of M1 money supply data, index data, and
    constituent stock data, combining them into a single dataset saved as a CSV file.

    Args:
        cfg (DictConfig): Configuration object containing parameters for data loading,
            including FRED series ID, date range, output file path, and ticker symbols.

    Attributes:
        cfg (DictConfig): Configuration object passed during initialization.
        series_id (str): FRED series ID for M1 money supply data.
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
        save_path (Path): Resolved file path for saving the output CSV file.
        index_ticker (str): Ticker symbol for the index data.
        constituent_ticker (str): Ticker symbol for the constituent stock data.
        crypto_ticker (str): Ticker symbol for the cryptocurrency data.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.series_id = cfg.load_data.fred_series_id
        self.start_date = cfg.load_data.start_date
        self.end_date = cfg.load_data.end_date
        self.save_path = cfg.load_data.output_file
        self.index_ticker = cfg.load_data.index_ticker
        self.constituent_ticker = cfg.load_data.constituent_ticker
        self.crypto_ticker = cfg.load_data.crypto_ticker

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

            index_df = data['Close'].rename(f'{self.index_ticker}_Close')
            index_df.index = pd.to_datetime(index_df.index)
            index_df.index.name = 'Date'

            logger.info("Loading existing M1 data from %s...", self.save_path)
            m1_df = pd.read_csv(self.save_path, index_col='Date', parse_dates=True)

            logger.info("Aligning and concatenating M1 data with index data...")
            combined_df = pd.concat([m1_df, index_df], axis=1, join='outer')
            logger.debug("Combined DataFrame head:\n%s", combined_df.head())

            combined_df.to_csv(self.save_path, index=True)
            logger.info("Appended index data to %s.", self.save_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error("An error occurred while processing the data: %s", e)
            raise

    def get_crypto_data(self):
        """
        Downloads cryptocurrency data, combines it with existing M1 data, and saves the result.

        This method retrieves historical cryptocurrency data from Yahoo Finance for the specified
        ticker and date range, aligns it with existing M1 data from a CSV file, handles
        missing values, and saves the combined dataset back to the CSV file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the CSV file at `self.save_path` does not exist.
            pandas.errors.EmptyDataError: If the CSV file is empty or corrupt.

        Notes:
            - Missing values in the combined dataset are dropped before saving.
        """
        try:
            data = yf.download(self.crypto_ticker,
                                start=self.start_date,
                                end=self.end_date,
                                multi_level_index=False,
                                auto_adjust=True,
                                )
            logger.info("Retrieved %d records from Yahoo Finance for cryptocurrency: %s.",
                        len(data), self.crypto_ticker)

            crypto_df = data['Close'].rename(f'{self.crypto_ticker}_Close')
            crypto_df.index = pd.to_datetime(crypto_df.index)
            crypto_df.index.name = 'Date'

            logger.info("Loading existing M1 data from %s...", self.save_path)
            m1_df = pd.read_csv(self.save_path, index_col='Date', parse_dates=True)

            logger.info("Aligning and concatenating M1 and index data with cryptocurrency data...")
            combined_df = pd.concat([m1_df, crypto_df], axis=1, join='outer')
            logger.debug("Combined DataFrame head:\n%s", combined_df.head())

            combined_df.to_csv(self.save_path, index=True)
            logger.info("Appended cryptocurrency data to %s.", self.save_path)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error("An error occurred while processing the data: %s", e)
            raise

    def get_constituent_data(self):
        """
        Downloads constituent stock data, combines it with existing data, and saves the result.

        This method retrieves historical stock data from Yahoo Finance for the specified
        constituent ticker and date range, aligns it with existing M1 and index data from
        a CSV file, handles missing values, and saves the combined dataset back to the CSV file.

        Returns:
            None

        Raises:
            FileNotFoundError: If the CSV file at `self.save_path` does not exist.
            pandas.errors.EmptyDataError: If the CSV file is empty or corrupt.

        Notes:
            - The stock data is retrieved with `auto_adjust=True` to adjust for splits/dividends.
            - Missing values in the combined dataset are dropped before saving.
        """
        try:
            data = yf.download(self.constituent_ticker,
                                start=self.start_date,
                                end=self.end_date,
                                multi_level_index=False,
                                auto_adjust=True,
                                )
            logger.info("Retrieved %d records from Yahoo Finance for stock: %s.",
                        len(data), self.constituent_ticker)

            stock_df = data['Close'].rename(f'{self.constituent_ticker}_Close')
            stock_df.index = pd.to_datetime(stock_df.index)
            stock_df.index.name = 'Date'

            logger.info("Loading existing M1 data from %s...", self.save_path)
            m1_df = pd.read_csv(self.save_path, index_col='Date', parse_dates=True)

            logger.info("Aligning and concatenating M1 and index data with stock data...")
            combined_df = pd.concat([m1_df, stock_df], axis=1, join='outer')
            logger.debug("Combined DataFrame head:\n%s", combined_df.head())

            combined_df.index = pd.to_datetime(combined_df.index)
            combined_df.index.name = 'Date'
            start = pd.to_datetime(self.start_date)
            end = pd.to_datetime(self.end_date)
            month_starts = pd.date_range(start=start, end=end, freq='MS')
            combined_df = combined_df.reindex(combined_df.index.union(month_starts)).sort_index()
            combined_df.index.name = 'Date'
            for col in combined_df.columns:
                series = combined_df[col]
                interp = series.interpolate(method='time', limit_direction='both')
                month_mask = series.index.isin(month_starts) & series.isna()
                if month_mask.any():
                    series.loc[month_mask] = interp.loc[month_mask]
                combined_df[col] = series

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
    """
    Main entry point for the data loading process.

    This function initializes the `LoadData` class with the provided configuration
    and sequentially calls methods to retrieve M1 money supply data, index data,
    cryptocurrency data, and constituent stock data, combining them into a single CSV file.

    Args:
        cfg (Optional[DictConfig]): Configuration object containing parameters for
            data loading. If None, Hydra loads the configuration from the specified
            config path and name. Defaults to None.

    Returns:
        None
    """
    data_loader = LoadData(cfg)
    data_loader.get_m1_data()
    data_loader.get_index_data()
    data_loader.get_crypto_data()
    data_loader.get_constituent_data()

if __name__ == "__main__":
    main()
