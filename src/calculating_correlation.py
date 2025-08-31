"""
Module for calculating correlations between money supply and index data.

This module provides functionality to compute long-term and short-term correlations
between M1 money supply and a specified index's closing prices using pandas and numpy.
"""
from pathlib import Path
from typing import Optional
import logging
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import hydra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calculating_correlation")

class CalculateCorrelation:
    """
    Class to calculate correlations between money supply and index data.

    Attributes:
        cfg (DictConfig): Configuration object from Hydra.
        data_path (Path): Resolved path to the input CSV data file.
        df (pd.DataFrame): DataFrame containing the input data with Date index.
        money_supply_col (str): Column name for M1 money supply data.
        index_col_name (str): Column name for the index's closing price.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = Path(repo_root / cfg.load_data.output_file)
        self.data_path = data_path.resolve()
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        self.money_supply_col = 'M1_Money_Supply'
        self.index_col_name = f"{cfg.load_data.index_ticker}_Close"

    def compute_long_term_correlation(self):
        """
        Compute the average long-term correlation by year.

        Groups data by year, calculates Pearson correlation between money supply
        and index closing prices, applies Fisher z-transformation, and computes
        the average correlation.

        Returns:
            float: Average long-term correlation coefficient, or NaN if no valid data.
        """
        correlations = []
        z_score = None
        min_periods = self.cfg.correlation.min_periods
        for year, group in self.df.groupby(self.df.index.year):
            _unused = year
            if len(group) >= min_periods:
                r = group[self.money_supply_col].corr(group[self.index_col_name])
                correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s", r, z_score)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Long-Term Correlation: %s", avg_r)

        return avg_r

    def compute_short_term_correlation(self):
        """
        Compute the average short-term correlation by quarter.

        Groups data by year and quarter, calculates Pearson correlation between
        money supply and index closing prices, applies Fisher z-transformation,
        and computes the average correlation.

        Returns:
            float: Average short-term correlation coefficient, or NaN if no valid data.
        """
        correlations = []
        z_score = None
        for (year, quarter), group in self.df.groupby([self.df.index.year, self.df.index.quarter]):
            _unused = year, quarter
            if len(group) < 3:
                continue
            r = group[self.money_supply_col].corr(group[self.index_col_name])
            correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s", r, z_score)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Short-Term Correlation: %s", avg_r)

        return avg_r

@hydra.main(version_base=None, config_path="../configs", config_name="calculating_correlation")
def main(cfg: Optional[DictConfig]=None):
    """
    Main function to compute long-term and short-term correlations.

    Instantiates CalculateCorrelation and calls methods to compute correlations.

    Args:
        cfg: Hydra configuration object, defaults to None.
    """
    calculator = CalculateCorrelation(cfg)
    calculator.compute_long_term_correlation()
    calculator.compute_short_term_correlation()

if __name__ == "__main__":
    main()
