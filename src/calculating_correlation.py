"""
Calculates correlations between M1 money supply and index closing prices.

Provides functions and classes to compute long-term and short-term correlations
using pandas and numpy. Supports time-lagged analysis and visualization.

Dependencies:
    pandas, numpy, matplotlib, hydra, omegaconf
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
    Calculates correlations between M1 money supply and index closing prices.

    Supports time-lagged analysis and computes long-term and short-term correlations
    using pandas and numpy.

    Attributes:
        cfg (DictConfig): Hydra configuration object.
        data_path (pathlib.Path): Resolved path to the input CSV data file.
        df (pandas.DataFrame): DataFrame with input data, indexed by Date.
        money_supply_col (str): Column name for M1 money supply data.
        col_name (str): Column name for the index or crypto closing price.
        min_periods (list): Minimum periods for correlation calculations.
        lag_period (int): Number of periods to shift data for lag analysis.
        time_lag (bool): Whether a time lag has been applied to the data.
    """
    def __init__(self, cfg: DictConfig, lag: int, column):
        super().__init__()
        self.cfg = cfg
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = Path(repo_root / cfg.load_data.output_file)
        self.data_path = data_path.resolve()
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        self.money_supply_col = 'M1_Money_Supply'
        self.col_name = column
        self.min_periods = cfg.correlation.min_periods
        self.lag_period = lag
        self.time_lag = False

    def apply_time_lag(self):
        """
        Applies a time lag to the specified column in the DataFrame.

        Shifts the values in the column specified by `col_name` by `lag_period` periods,
        removes rows with missing values, and sets `time_lag` to True. For crypto data,
        removes rows before 2014-09-01.

        Returns:
            None
    """
        if self.col_name == f"{self.cfg.load_data.crypto_ticker}_Close":
            # remove all rows before 2014-09-01 for the crypto data
            cutoff = pd.to_datetime("2014-09-01")
            self.df = self.df.loc[self.df.index >= cutoff]
        self.df[self.col_name] = self.df[self.col_name].shift(self.lag_period)
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
                r = group[self.money_supply_col].corr(group[self.col_name])
                correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s for %s", r, z_score, self.col_name)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Long-Term Correlation (%s months) for %s: %s",
                    self.lag_period, self.col_name, avg_r)
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
            r = group[self.money_supply_col].corr(group[self.col_name])
            correlations.append(r)
        z_score = np.arctanh(correlations)
        logger.debug("Correlation: %s, Z-Score: %s for %s", r, z_score, self.col_name)
        avg_z = np.mean(z_score) if z_score.size > 0 else np.nan
        avg_r = np.tanh(avg_z) if not np.isnan(avg_z) else np.nan
        logger.info("Average Short-Term Correlation (%s months) for %s: %s",
                    self.lag_period, self.col_name, avg_r)

        return avg_r

@hydra.main(version_base=None, config_path="../configs", config_name="calculating_correlation")
def main(cfg: Optional[DictConfig]=None):
    """
    Computes and visualizes long-term and short-term correlations.

    Instantiates `CalculateCorrelation` for specified lag periods, computes correlations
    between M1 money supply and index/crypto closing prices, and visualizes results in a
    bar chart using matplotlib.

    Args:
        cfg (Optional[omegaconf.DictConfig]): Hydra configuration object. Defaults to None.

    Returns:
        None
    """
    long_term_corr = {}
    short_term_corr = {}

    index_col = f"{cfg.load_data.index_ticker}_Close"
    crypto_col = f"{cfg.load_data.crypto_ticker}_Close"
    for column in [index_col, crypto_col]:
        long_term_corr[column] = []
        short_term_corr[column] = []
        for lag in cfg.correlation.lag_periods:
            calculator = CalculateCorrelation(cfg, lag, column)
            calculator.apply_time_lag()
            long_term_corr[column].append(calculator.compute_long_term_correlation())
            short_term_corr[column].append(calculator.compute_short_term_correlation())

    # Plotting: Create two subplots, one for each column
    x = np.arange(len(cfg.correlation.lag_periods))
    width = 0.25
    bar_colours = ["#1f77b4", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), layout='constrained')
    _unused = fig

    axes = {index_col: ax1, crypto_col: ax2}

    for column, ax in axes.items():
        long_term_data = long_term_corr[column]
        short_term_data = short_term_corr[column]


        rects1 = ax.bar(x - width/2, long_term_data, width, label="Long Term", color=bar_colours[0])

        rects2 = ax.bar(x + width/2, short_term_data, width, label="Short Term", color=bar_colours[1])

        # Add bar labels
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        # Set labels and title
        ax.set_ylabel("Correlation Coefficient")
        ax.set_xlabel("Lag Periods")
        ax.set_xticks(x, labels=[f"{lag} months" for lag in cfg.correlation.lag_periods])
        ax.set_title(f"Correlation Coefficients for {column}")
        ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
