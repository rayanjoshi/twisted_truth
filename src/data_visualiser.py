"""
Visualize financial data including M1 money supply and stock prices.

This module provides the `DataVisualiser` class to load financial data from a CSV file
and create a plot to visualize the relationship between M1 money supply, an index ticker,
and a constituent stock ticker's close prices over time. It uses Hydra for configuration
management and pandas/matplotlib for data handling and visualization.

Dependencies:
    - pathlib: For file path handling.
    - pandas: For data loading and manipulation.
    - matplotlib: For creating plots.
    - omegaconf: For configuration management.
    - hydra: For configuration loading and application execution.
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra

class DataVisualiser:
    """
    A class to visualize financial data, including M1 money supply and stock prices.

    This class loads data from a CSV file specified in the configuration and creates
    a plot to visualize the relationship between M1 money supply and stock prices
    for a given index and constituent ticker.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        data_path = Path(repo_root / cfg.load_data.output_file)
        self.data_path = data_path.resolve()
        self.df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        self.index_col_name = f"{cfg.load_data.index_ticker}_Close"
        self.stock_col_name = f"{cfg.load_data.constituent_ticker}_Close"

    def plot_data(self):
        """
        Create a plot of M1 money supply and stock prices.

        Generates a matplotlib figure with a primary axis for M1 money supply
        and twin axes for the index and constituent stock prices. The plot
        visualizes the relationship between these financial metrics over time.
        """
        fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
        fig.subplots_adjust(hspace=0.05)  # Small space between subplots


        ax1.plot(self.df.index,
                self.df['M1_Money_Supply'],
                color='tab:blue',
                label='M1 Money Supply',
                )
        ax2_index = ax1.twinx()
        ax2_stock = ax1.twinx()
        # Plot Index and Stock on ax2_index (higher range) and ax2_stock (lower range)
        # Top axis for index (higher values, e.g., ~2900 and above)
        ax2_index.plot(self.df.index,
                    self.df[self.index_col_name],
                    color='tab:orange',
                    label=f'{self.cfg.load_data.index_ticker} Close Price',
                    )
        # Bottom axis for stock (lower values, e.g., ~0 to 250)
        ax2_stock.plot(self.df.index,
                        self.df[self.stock_col_name],
                        color='tab:green',
                        label=f'{self.cfg.load_data.constituent_ticker} Close Price',
                        )

        # Labels and ticks
        ax1.set_xlabel('Date')
        ax1.set_ylabel('M1 Money Supply / US$')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2_index.set_ylabel('Close Price / US$')
        ax2_index.tick_params(axis='y', labelcolor='tab:orange')
        ax2_stock.tick_params(axis='y', labelcolor='tab:green')

        # Add legends
        ax1.legend(loc='lower left')
        ax2_index.legend(loc='lower center')
        ax2_stock.legend(loc='lower right')

        plt.title('Relationship Between M1 Money Supply and Stock Prices')
        plt.show()

@hydra.main(version_base=None, config_path="../configs", config_name="data_visualiser")
def main(cfg: Optional[DictConfig]=None):
    """
    Main function to initialize and run the data visualizer.

    Args:
        cfg (Optional[DictConfig], optional): Configuration object for data loading
            and visualization. Defaults to None, in which case Hydra loads the
            configuration from the specified path and name.
    """
    visualizer = DataVisualiser(cfg)
    visualizer.plot_data()

if __name__ == "__main__":
    main()
