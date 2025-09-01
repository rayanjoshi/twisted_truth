"""
Module for calculating correlations between money supply and index data.

This module provides functionality to compute long-term and short-term correlations
between M1 money supply and a specified index's closing prices using pandas and numpy.
"""
from pathlib import Path
from typing import Optional
import logging
import matplotlib.pyplot as plt
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
        lag_period (int): Number of periods to shift the index data for lag analysis.
        time_lag (bool): Indicates whether a time lag has been applied to the data.
    """
    def __init__(self, cfg: DictConfig, lag: int):
        super().__init__()
        self.cfg = cfg
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = Path(repo_root / cfg.load_data.output_file)
        self.data_path = data_path.resolve()
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        self.money_supply_col = 'M1_Money_Supply'
        self.index_col_name = f"{cfg.load_data.index_ticker}_Close"
        self.min_periods = cfg.correlation.min_periods
        self.lag_period = lag
        self.time_lag = False

    def apply_time_lag(self):
        """
        Apply a time lag to a specified column in the DataFrame.

        Shifts the values in the column specified by `index_col_name` by `lag_period` 
        periods, removes rows with missing values resulting from the shift, and sets 
        the `time_lag` attribute to True.
        """
        self.df[self.index_col_name] = self.df[self.index_col_name].shift(self.lag_period)
        self.df.dropna(inplace=True)
        self.time_lag = True

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
        z_score = np.array([])
        for year, group in self.df.groupby(self.df.index.year):
            _unused = year
            if len(group) >= self.min_periods[0]:
                r = group[self.money_supply_col].corr(group[self.index_col_name])
                correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s", r, z_score)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Long-Term Correlation (%s months): %s", self.lag_period, avg_r)
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
        z_score = np.array([])
        for (year, quarter), group in self.df.groupby([self.df.index.year, self.df.index.quarter]):
            _unused = year, quarter
            if len(group) < self.min_periods[1]:
                continue
            r = group[self.money_supply_col].corr(group[self.index_col_name])
            correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s", r, z_score)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Short-Term Correlation (%s months): %s", self.lag_period, avg_r)

        return avg_r

@hydra.main(version_base=None, config_path="../configs", config_name="calculating_correlation")
def main(cfg: Optional[DictConfig]=None):
    """
    Main function to compute long-term and short-term correlations.

    Instantiates CalculateCorrelation, computes correlations for specified lag periods,
    and visualizes results in a bar chart.

    Args:
        cfg (Optional[DictConfig]): Hydra configuration object, defaults to None.
    """
    long_term_corr = []
    short_term_corr = []
    for lag in cfg.correlation.lag_periods:
        calculator = CalculateCorrelation(cfg, lag)
        calculator.apply_time_lag()
        long_term_corr.append(calculator.compute_long_term_correlation())
        short_term_corr.append(calculator.compute_short_term_correlation())

    x = np.arange(len(cfg.correlation.lag_periods))
    width = 0.25
    multiplier = 0
    corr_dict = {"Long Term": long_term_corr, "Short Term": short_term_corr}
    fig, ax = plt.subplots(layout='constrained', figsize=(8, 6))
    _unused = fig
    bar_colours = ["#1f77b4", "#d62728"]
    for attribute, measurement in corr_dict.items():
        offset = width * multiplier
        # set the bar facecolor per group and use a contrasting label color
        rects = ax.bar(x + offset,
                        measurement,
                        width,
                        label=attribute,
                        color=bar_colours[multiplier]
                        )
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_ylabel("Correlation Coefficient")
    ax.set_xlabel("Lag Periods")
    ax.set_xticks(x + width, labels=[f"{lag} months" for lag in cfg.correlation.lag_periods])
    ax.set_title("Correlation Coefficients by Lag Period")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
