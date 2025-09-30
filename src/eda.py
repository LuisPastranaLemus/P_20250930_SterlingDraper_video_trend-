# Exploratory Data Analysis for Visualizations and summary statistics

from IPython.display import display, HTML
from itertools import combinations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import t, laplace, norm, shapiro
import scipy.stats as stats


# Function to detect outlier boundaries with optional clamping of lower bound to zero
def outlier_limit_bounds(df, column, bound='both', clamp_zero=False):
    """
    Detects outlier thresholds based on the IQR method and returns rows beyond those limits.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - column (str): The name of the numerical column to analyze.
    - bound (str): One of 'both', 'lower', or 'upper' to indicate which bounds to evaluate.
    - clamp_zero (bool): If True, clamps the lower bound to zero (useful for non-negative metrics).

    Returns:
    DataFrame(s): Rows identified as outliers, depending on the bound selected.
    """

    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1

    lower_bound = max(q1 - 1.5 * iqr, 0) if clamp_zero else q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    display(HTML(f"> Outlier thresholds for <i>'{column}'</i>: \n"
                 f"> Lower = <b>{lower_bound:.3f}</b>, > Upper = <b>{upper_bound:.3f}</b>"))

    if bound not in ['both', 'lower', 'upper']:
        display(HTML(f"> Invalid 'bound' parameter. Use <b>'both'</b>, <b>'upper'</b>, or <b>'lower'</b>."))
        return

    outliers = pd.DataFrame()
    
    if bound in ['both', 'lower']:
        lower_outliers = df[df[column] < lower_bound]
        if lower_outliers.empty:
            display(HTML(f"> <b>No</b> lower outliers found in column <i>'{column}'</i>."))
        outliers = pd.concat([outliers, lower_outliers])

    if bound in ['both', 'upper']:
        upper_outliers = df[df[column] > upper_bound]
        if upper_outliers.empty:
            display(HTML(f"> <b>No</b> upper outliers found in column <i>'{column}'</i>."))
        outliers = pd.concat([outliers, upper_outliers])

    display(HTML(f"- - -"))
    display(HTML(f"> Outliers:"))

    return outliers if not outliers.empty else None

# Function to evaluate the central tendency of a numerical feature
def evaluate_central_trend(df, column):
    """
    Evaluates the central tendency of a given column using the coefficient of variation (CV)
    and skewness to determine the most reliable measure (mean or median).
    
    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): Name of the numerical column to evaluate.
    
    Output:
    Displays CV, skewness, and recommends the most reliable central measure.
    """
    
    data = df[column].dropna()
    
    if data.empty:
        display(HTML(f"<b>Column '{column}' is empty or contains only NaNs.</b>"))
        return
    
    mean = data.mean()
    std = data.std()
    skew = data.skew()
    
    if mean == 0:
        display(HTML(f"> Mean of column '{column}' is <b>zero</b>.\n Coefficient of Variation is <b>undefined</b>."))
        return
    
    cv = (std / mean) * 100
    
    # CV-based interpretation
    if cv <= 10:
        cv_msg = "Very low variability: highly reliable mean."
    elif cv <= 20:
        cv_msg = "Moderate variability: reasonably reliable mean."
    elif cv <= 30:
        cv_msg = "Considerable variability: mean may be biased."
    else:
        cv_msg = "High variability: mean may be misleading."
    
    # Skewness-based adjustment
    abs_skew = abs(skew)
    if abs_skew <= 0.3:
        skew_msg = "Low skewness: distribution is nearly symmetric."
        skew_level = "low"
    elif abs_skew <= 0.6:
        skew_msg = "Moderate skewness: some asymmetry present."
        skew_level = "moderate"
    elif abs_skew <= 1.0:
        skew_msg = "High skewness: strong asymmetry detected."
        skew_level = "high"
    else:
        skew_msg = "Very high skewness: distribution is heavily distorted."
        skew_level = "very_high"
    
    # Central trend evaluation
    if cv > 30 or skew_level in ["high", "very_high"]:
        central = "median"
        reason = "due to high variability or strong skewness"
    elif cv > 20 or skew_level == "moderate":
        central = "median (with caution)"
        reason = "due to moderate variability or skewness"
    else:
        central = "mean"
        reason = "distribution is stable and symmetric"

    display(HTML(f"> Coefficient of variation for column <i>'{column}'</i>: <b>{cv:.2f} %</b>"))
    display(HTML(f"> Skewness of column <i>'{column}'</i>: <b>{skew:.2f}</b>"))
    display(HTML(f"> {cv_msg}"))
    display(HTML(f"> {skew_msg}"))
    display(HTML(f"> Recommended central measure: <b>{central}</b> ({reason})"))
    
    # Validation of values for transformation
    min_val = data.min()
    has_negatives = (min_val < 0)
    has_zeros = (data == 0).any()
    all_positive = (min_val > 0)

    # Robust Transformation Suggestion
    if skew_level in ["high", "very_high"]:
        if skew > 0:
            transform_suggestion = "To reduce right skew:"
            if all_positive:
                transform_suggestion += " [log(x), sqrt(x), reciprocal(x), Box-Cox]."
            elif not has_negatives:
                transform_suggestion += " [sqrt(x), reciprocal(x), Yeo-Johnson (handles zeros)]."
            else:
                transform_suggestion += " [Yeo-Johnson, quantile or rank-based transforms (handles negatives)]."
        else:
            transform_suggestion = "To reduce left skew:"
            if not has_negatives:
                transform_suggestion += " [square(x), exp(x), reflect+log(x), Yeo-Johnson]."
            else:
                transform_suggestion += " [Yeo-Johnson or rank-based transforms (handles negatives)]."

        if abs_skew > 1.5 or data.max() > 10 * data.median():
            transform_suggestion += " For extreme skew or heavy-tailed distributions, consider quantile or normal score transforms instead of classical ones."

        display(HTML(f"> Suggested transformation: <i>{transform_suggestion}</i>"))
    
    return 

# function in order to select the correct bins calculation
# bins = calculate_bins(df['total_spent'], method='fd')
def calculate_bins(data, method='fd'):
    """
    Calculate optimal number of histogram bins using a selected method.

    Parameters:
    - data (array-like): Input data for which to calculate bins.
    - method (str): Method for calculating bins. Options:
        'sturges', 'sqrt', 'fd' (Freedman–Diaconis), 'rice', 'auto'

    Returns:
    - int: Number of bins.
    """
    data = np.asarray(data.dropna())  # ensure no NaNs

    n = len(data)
    if n == 0:
        raise ValueError("Data must contain at least one non-NaN value.")

    if method == 'sturges':
        return int(np.ceil(np.log2(n) + 1))
    elif method == 'sqrt':
        return int(np.sqrt(n))
    elif method == 'rice':
        return int(2 * n ** (1/3))
    elif method == 'fd':
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / n ** (1/3)
        return int((data.max() - data.min()) / bin_width)
    elif method == 'auto':
        # matplotlib can handle 'auto' directly
        return 'auto'
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'sturges', 'sqrt', 'fd', 'rice', 'auto'.")


# Function to evaluate pairwise correlations among numerical columns
def evaluate_correlation(df, columns=None):
    """
    Evaluates pairwise Pearson correlations between numerical columns in a DataFrame.
    
    Parameters:
    - df (DataFrame): Input DataFrame with at least two numeric columns.
    
    Output:
    - Displays correlation coefficients with interpretation:
        > Strong correlation (|r| > 0.7)
        > Moderate correlation (0.3 < |r| ≤ 0.7)
        > Weak or no linear relationship (|r| ≤ 0.3)
        > Positive vs. Negative direction
    """

    if columns is None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_cols:
            raise ValueError("No hay columnas numéricas en la lista especificada.")

    seen_pairs = set()

    for col_x in numeric_cols:
        for col_y in numeric_cols:
            if col_x != col_y and (col_y, col_x) not in seen_pairs:
                corr = df[col_x].corr(df[col_y])
                abs_corr = abs(corr)

                strength = ''
                direction = 'positive' if corr > 0 else 'negative' if corr < 0 else 'neutral'
                if abs_corr > 0.7:
                    strength = 'Strong'
                elif abs_corr > 0.3:
                    strength = 'Moderate'
                elif abs_corr == 0:
                    strength = 'No linear relationship'
                else:
                    strength = 'Weak'

                if abs_corr > 0:
                    if strength in ['Strong', 'Moderate']:  
                        display(HTML(f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                                     f"<b>{strength} {direction} correlation</b><br><br>"))
                    else:
                        display(HTML(f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                                     f"{strength} {direction} correlation<br><br>"))
                else:
                    if strength in ['Strong', 'Moderate']: 
                        display(HTML(f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                                     f"<b>{strength}</b><br><br>"))
                    else:
                        display(HTML(f"> Correlation (<i>{col_x}</i>, <i>{col_y}</i>): <b>{corr:.2f}</b><br>"
                                     f"{strength}<br><br>"))
                        
                seen_pairs.add((col_x, col_y))

# Function to visualize missing values within a DataFrame using a heatmap
def missing_values_heatmap(df):
    """
    Displays a heatmap of missing (NaN) values in the given DataFrame.
    
    Parameters:
    -df (DataFrame): The input DataFrame to analyze.
    
    Output:
    A heatmap visualization showing the presence of missing values per column and row.
    """
    plt.figure(figsize=(15, 7))
    sns.heatmap(df.isna(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Values')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.show()

# Function for heatmap data visualization
# df['Region', 'Product', 'Sales']
# plot_heatmap(df, title='Sales Heatmap by Region and Product', xlabel='Region', ylabel='Product', cmap='YlOrRd', fmt='.0f',
#              cbar_label='Sales Volume')   
def plot_heatmap(data, title='', xlabel='', ylabel='', cmap='YlGnBu', annot=True, fmt='d', cbar_label='', figsize=(15, 7)):
    """
    Plots a heatmap with customization options.

    Parameters:
    - data (DataFrame): Pivot table to visualize.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - cmap (str): Color map for heatmap.
    - annot (bool): Whether to show values inside the heatmap cells.
    - fmt (str): Format for annotation text 'd' for integer, '.0f' for float.
    - cbar_label (str): Label for the color bar.
    - figsize (tuple): Figure size (width, height).

    Returns:
    None: Displays the heatmap.
    """
       
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, cbar_kws={'label': cbar_label})

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

# Function to plot multiple boxplots side by side for comparison
# ds_list=[df1['column_name], df['column_name], df['column_name]]
# plot_boxplots(ds_list=[serie1, serie2, serie3], xlabels=['Group A', 'Group B', 'Group C'], ylabel='Values', 
#               title='Comparison of Value Distributions Across Groups', yticks_range=(0, 40, 5), rotation=45,
#               color=['skyblue', 'lightgreen', 'salmon']
def plot_boxplots(ds_list, xlabels, ylabel, title, yticks_range=None, rotation=0, color='grey'):
    """
    Plots multiple boxplots side by side, allowing for visual comparison across groups.

    Parameters:
    - ds_list (list of Series): List of numerical pandas Series to plot.
    - xlabels (list of str): Corresponding labels for each dataset.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.
    - yticks_range (tuple, optional): Range for y-axis ticks, e.g., (min, max, step).
    - rotation (int, optional): Rotation angle for x and y tick labels.
    - color (str or list, optional): Either a single color or a list of colors matching the groups.

    Raises:
    ValueError: If the number of datasets and labels do not match.

    Output:
    Displays a customized boxplot figure for group-wise value comparison.
    """

    if len(ds_list) != len(xlabels):
        raise ValueError("*** Error *** > The data list and labels must be the same length.")
    
    df = pd.DataFrame({
        'value': pd.concat(ds_list, ignore_index=True),
        'group': sum([[label] * len(s) for label, s in zip(xlabels, ds_list)], [])
    })

    plt.figure(figsize=(15, 7))

    # If color is a list, assign a custom palette; if string, use a solid color
    if isinstance(color, (list, tuple)) and len(color) == len(xlabels):
        palette = dict(zip(xlabels, color))
        sns.boxplot(x='group', y='value', hue='group', data=df, palette=palette)
    else:
        sns.boxplot(x='group', y='value', data=df, color=color)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)

    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a histogram with mean and median reference lines
# ds | df['column_name']
# plot_histogram(df_product_purchase_quantity['total_products'], bins=range(0, 65, 1), color='grey', title='Distribution for Product Quantity by Orders',
#                xlabel='Products', ylabel='Frequency', xticks_range=range(0, 65, 5), yticks_range=range(0, 200, 20), rotation=45)
def plot_histogram(ds, bins=10, color='grey', title='', xlabel='', ylabel='Frequency', xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots a histogram for a given numerical Series with optional customization.

    Parameters:
    - ds (Series): The numerical data to plot.
    - bins (int or array-like): Number or range of histogram bins.
    - color (str): Fill color for the bars.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xticks_range (range, optional): Range object for x-ticks (e.g., range(0, 100, 10)).
    - yticks_range (range, optional): Range object for y-ticks (e.g., range(0, 10, 1)).
    - rotation (int): Angle of tick label rotation.

    Output:
    Displays a histogram with vertical lines for mean and median.
    """

    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, edgecolor='black', color=color, kde=False)

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if isinstance(xticks_range, range):
        plt.xlim(xticks_range.start, xticks_range.stop)
        plt.xticks(xticks_range, rotation=rotation)

    if isinstance(yticks_range, range):
        plt.ylim(yticks_range.start, yticks_range.stop)
        plt.yticks(yticks_range, rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a stacked histogram by group (hue = category)
# df['column_name', 'column_categorical'] column_categorical = different values for hue
# plot_hue_histogram(df, x_col='amount', hue_col='sex', bins=20, title='Distribution for bill by gender',
#                    xlabel='Totla_bill ($)', ylabel='Frequency', legend_title='Gender', legend_labels=['Male', 'Female'])
def plot_hue_histogram(df, x_col='', hue_col='', bins=30, color='grey', title='', xlabel='', ylabel='',
                       legend_title='', legend_labels=[]):
    """
    Plots a stacked histogram with grouping by a categorical variable (hue).

    Parameters:
    - df (DataFrame): Input dataset.
    - x_col (str): Numerical column to plot on the x-axis.
    - hue_col (str): Categorical column used to group data.
    - bins (int): Number of histogram bins.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - legend_title (str): Title for the legend.
    - legend_labels (list, optional): Custom labels for legend categories.

    Output:
    Displays a stacked histogram with hue-based grouping.
    """
    
    plt.figure(figsize=(15, 7))
    sns.histplot(data=df, x=x_col, hue=hue_col, multiple='stack', bins=bins, color=color)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if legend_labels:
        plt.legend(title=legend_title, labels=legend_labels)
    else:
        plt.legend(title=legend_title)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to compare two distributions using overlapping histograms
# male_bills = tips[tips['sex'] == 'Male']['total_bill']
# female_bills = tips[tips['sex'] == 'Female']['total_bill']
# plot_dual_histogram(ds1=male_bills, ds2=female_bills, bins=15, color1='black', color2='grey', title='Comparison of Total Bill Distribution by Gender',
#                     xlabel='Total Bill ($)', ylabel='Frequency', label1='Male', label2='Female', xticks_range=(0, 60, 5), 
#                     yticks_range=(0, 80, 10), rotation=45)
def plot_dual_histogram(ds1, ds2, bins=10, color1='black', color2='grey', title='Histogram Comparison', xlabel='', ylabel='',
                        label1='', label2='', xticks_range=None, yticks_range=None, rotation=0):
    """
    Plots two overlapping histograms to visually compare distributions.

    Parameters:
    - ds1 (Series): First numerical dataset.
    - ds2 (Series): Second numerical dataset.
    - bins (int): Number of bins for the histogram.
    - color1 (str): Color for the first dataset.
    - color2 (str): Color for the second dataset.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - label1 (str): Legend label for the first dataset.
    - label2 (str): Legend label for the second dataset.
    - xticks_range (tuple, optional): Range and step for x-ticks (min, max, step).
    - yticks_range (tuple, optional): Range and step for y-ticks (min, max, step).
    - rotation (int): Tick label rotation angle.

    Output:
    Displays overlapping histograms with mean and median lines for both datasets.
    """

    # Clean missing values
    ds1 = ds1.dropna()
    ds2 = ds2.dropna()

    # Compute statistics
    mean1_val = ds1.mean()
    median1_val = ds1.median()
    mean2_val = ds2.mean()
    median2_val = ds2.median()

    plt.figure(figsize=(15, 7))

    sns.histplot(ds1, bins=bins, edgecolor='black', kde=False, color=color1, label=label1, alpha=0.8)
    sns.histplot(ds2, bins=bins, edgecolor='black', kde=False, color=color2, label=label2, alpha=0.6)

    plt.axvline(mean1_val, color='red', linestyle='dashed', linewidth=1.5, label=f'{label1} Mean: {mean1_val:.2f}')
    plt.axvline(mean2_val, color='darkred', linestyle='dashed', linewidth=1.5, label=f'{label2} Mean: {mean2_val:.2f}')
    plt.axvline(median1_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'{label1} Median: {median1_val:.2f}')
    plt.axvline(median2_val, color='darkblue', linestyle='dashdot', linewidth=1.5, label=f'{label2} Median: {median2_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range is not None:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)
    if yticks_range is not None:
        plt.ylim(yticks_range[0], yticks_range[1])
        plt.yticks(np.arange(*yticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a frequency density histogram with optional KDE overlay
# ds | df['column_name']
# plot_frequency_density(tips['tip'], bins=20, color='grey', title='Density Plot for Tips', xlabel='Tip Amount ($)', ylabel='Density',
#                        xticks_range=(0, 11, 1), rotation=45, show_kde=True)
def plot_frequency_density(ds, bins=10, color='grey', title='', xlabel='', ylabel='Density',
                           xticks_range=None, rotation=0, show_kde=True):
    """
    Plots a frequency density histogram with optional KDE curve.

    Parameters:
    - ds (Series): Numerical data to plot.
    - bins (int or array-like): Number or range of bins for the histogram.
    - color (str): Histogram bar color.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis (default: 'Density').
    - xticks_range (tuple, optional): Tuple (min, max, step) for x-tick configuration.
    - rotation (int, optional): Angle for tick label rotation.
    - show_kde (bool, optional): Whether to overlay a KDE curve.

    Output:
    Displays a histogram normalized to show frequency density, with mean/median lines and optional KDE.
    """

    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    plt.figure(figsize=(15, 7))
    sns.histplot(ds, bins=bins, stat='density', edgecolor='black', color=color, alpha=0.7)

    if show_kde:
        sns.kdeplot(ds, color='darkblue', linewidth=2, label='KDE')

    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='blue', linestyle='dashdot', linewidth=1.5, label=f'Median: {median_val:.2f}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xticks_range:
        plt.xlim(xticks_range[0], xticks_range[1])
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a grouped barplot (categorical x-axis, grouped by hue)
# df.groupby(['day', 'sex'])['tip'].mean().reset_index()
# df['day', 'sex', 'tip']
# plot_grouped_barplot(ds=dataframe, x_col='month', y_col='median_duration', hue_col='plan', palette=['black', 'grey'],
#                      title='Average Call Duration by Plan and Month', xlabel='Month', ylabel='Average Call Duration (min)',
#                      xticks_range=range(0, 13, 1), yticks_range=range(0, 500, 50), rotation=65)

# plot_hue_barplot(df, x_col='day', y_col='tip', hue_col='sex', title='Average Tip by Day and Gender', xlabel='Day of Week', 
#                  ylabel='Average Tip ($)', xticks_range=range(0, 13, 1), yticks_range=range(0, 500, 50), x_rotation=0, y_rotation=0, 
#                  alpha=0.95, show_legend=True, show_values=True)
def plot_hue_barplot(ds, x_col, y_col, hue_col=None, palette=sns.color_palette("PRGn", n_colors=50), title='', xlabel='', ylabel='', 
                         xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0, alpha=0.95, show_legend=True, show_values=True):
    """
    Plots a grouped bar chart with categorical grouping (hue).

    Parameters:
    - ds (DataFrame): The dataset to use for plotting.
    - x_col (str): The column to use for the x-axis (categorical).
    - y_col (str): The column to plot as the bar height (numerical).
    - hue_col (str, optional): The column to group by within each x-category.
    - palette (list, optional): List of colors for each hue category.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xticks_range (range, optional): Tick range and step for the x-axis.
    - yticks_range (range, optional): Tick range and step for the y-axis.
    - x_rotation (int): Rotation angle for x-axis ticks.
    - y_rotation (int): Rotation angle for y-axis ticks.
    - alpha (float): Transparency of bars.
    - show_legend (bool): Whether to display the legend (default: True).
    - show_values (bool): Whether to display value labels on top of bars.

    Output:
    Displays a grouped bar plot with optional axis customization and legend.
    """

    fig, ax = plt.subplots(figsize=(15, 7))
    palette = palette
    strong_palette = palette[:13] + palette[-12:]
    sns.barplot(data=ds, x=x_col, y=y_col, hue=hue_col, palette=strong_palette, alpha=alpha, ax=ax)

    if show_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            height + (0.01 * height),
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks_range is not None:
        ax.set_xticks(ticks=xticks_range)
    ax.tick_params(axis='x', rotation=x_rotation)

    if yticks_range is not None:
        ax.set_yticks(ticks=yticks_range)
    ax.tick_params(axis='y', rotation=y_rotation)

    if not show_legend:
        ax.legend().remove()

    ax.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a horizontal bar chart from categorical data
# ds | df['column_name']
# plot_horizontal_bar(df['sex'], colors=['#4c72b0', '#dd8452'], xlabel='Number of Records', ylabel='Gender', title='Count of Records by Gender',
#                     xticks_range=(0, 150, 10), rotation=0, show_values=True)
def plot_categorical_horizontal_bar(ds, colors=['black', 'grey'], xlabel='', ylabel='', title='', xticks_range=None, rotation=0, show_values=True):
    """
    Plots a horizontal bar chart for a categorical pandas Series.

    Parameters:
    ds (Series): Categorical data to summarize and visualize.
    colors (list): Color palette for each category.
    xlabel (str): Label for the x-axis (typically counts).
    ylabel (str): Label for the y-axis (categories).
    title (str): Title of the plot.
    xticks_range (tuple, optional): Tuple (min, max, step) for x-axis ticks.
    rotation (int): Rotation angle for x-axis tick labels.
    show_values (bool): Whether to display the value at the end of each bar.

    Output:
    Displays a horizontal bar chart with optional hue differentiation.
    """

    categories = ds.value_counts().index
    values = ds.value_counts().values

    plt.figure(figsize=(15, 7))
    ax = sns.barplot(y=categories, x=values, hue=categories, dodge=False, palette=colors)
    
    if show_values:
        for i, v in enumerate(values):
            ax.text(v + max(values)*0.01, i, f'{v}', va='center', fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks_range is not None:
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Function to plot a horizontal bar chart
# ds 1 df['column_name']
# avg_tip_by_day = tips.groupby('day')['tip'].mean().sort_values()
# plot_horizontal_numeric_bars(avg_tip_by_day, xlabel='Average Tip ($)', ylabel='Day', title='Average Tip by Day', color='seagreen',
#                              xticks_range=(0, 4.5, 0.5), show_values=True)
def plot_horizontal_bar(ds, xlabel='', ylabel='', title='', color='steelblue', xticks_range=None, rotation=0, show_values=True):
    """
    Plots a horizontal bar chart for numerical values.

    Parameters:
    - data (Series or DataFrame): A Series with index as labels and values as numeric data.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - title (str): Title of the plot.
    - color (str): Color of the bars.
    - xticks_range (tuple, optional): (min, max, step) for x-ticks.
    - rotation (int): Rotation angle for x-axis tick labels.
    - show_values (bool): Whether to display values at end of each bar.
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()

    labels = data.index
    values = data.values

    plt.figure(figsize=(15, 7))
    ax = sns.barplot(y=labels, x=values, color=color)

    if show_values:
        for i, v in enumerate(values):
            ax.text(v + max(values)*0.01, i, f'{v:.2f}', va='center', fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xticks_range:
        plt.xticks(np.arange(*xticks_range), rotation=rotation)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot grouped bar charts from a DataFrame with multiple columns
# df['column1_name', 'column2_name']
# avg_data = tips.groupby('day')[['total_bill', 'tip']].mean().round(2)
# plot_grouped_bars(avg_data, title='Average Total Bill and Tip by Day', xlabel='Day of Week', ylabel='Amount ($)', x_rotation=0, y_rotation=0,
#                  grid_axis='y', color=['#1f77b4', '#ff7f0e'], show_values=True)
def plot_grouped_bars(df, title='', xlabel='', ylabel='', x_rotation=0, y_rotation=0, grid_axis='y', color='grey', show_values=True):
    """
    Plots grouped (clustered) bar charts for comparing multiple values across an "index".

    Parameters:
    - df (DataFrame): A DataFrame where the index defines groups and columns.
    - title (str): Title of the chart.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - x_rotation (int): Rotation angle for x-axis tick labels.
    - y_rotation (int): Rotation angle for y-axis tick labels.
    - grid_axis (str): Axis along which to display grid lines ('x', 'y', or 'both').
    - show_values (bool): Whether to display values on top of each bar.

    Output:
    Displays a grouped bar chart comparing values across index categories and columns.
    """

    ax = df.plot(kind='bar', figsize=(15, 7), color=color)
    
    if show_values:
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + (0.01 * height),
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=x_rotation)
    plt.yticks(rotation=y_rotation)
    plt.grid(axis=grid_axis)
    plt.tight_layout()
    plt.show()

# Function to generate a customizable seaborn pairplot for exploratory correlation analysis (COV)
# df | df['column1_name', 'column2_name', ..., 'columnN_name']
# plot_pairplot(df, height=2.5, aspect=1.5, point_color='slategray')
def plot_pairplot(df, height=3, aspect=2.5, point_color='grey'):
    """
    Plots a Seaborn pairplot for all numeric columns in a DataFrame.

    Parameters:
    - df (DataFrame): The dataset to plot.
    - height (float): Height (in inches) of each facet (subplot).
    - aspect (float): Aspect ratio of each facet (width = height × aspect).

    Returns:
    None: Displays the pairplot.
    """
    sns.pairplot(df, height=height, aspect=aspect, plot_kws={'color': point_color})
    plt.tight_layout()
    plt.show()

# Function to plot a scatter matrix for exploring pairwise relationships (CORR)
# df | df['column1_name', 'column2_name', ..., 'columnN_name']
# plot_scatter_matrix(df, figsize=(12, 10), diagonal='kde', color='teal', alpha=0.4)
def plot_scatter_matrix(df, columns=None, figsize=(15, 7), color='grey', alpha=0.4):
    """
    Plots a lower-triangle scatter matrix with histograms on the diagonal and correlation annotations.

    Parameters:
    - df (DataFrame): Input dataset.
    - columns (list): Optional list of columns to include. If None, uses all numeric.
    - figsize (tuple): Size of the full figure.
    - color (str): Color of scatter points.
    - alpha (float): Transparency of points.

    Returns:
    - None: Displays the plot.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=figsize)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x = columns[j]
            y = columns[i]

            if i > j:
                sns.scatterplot(data=df, x=x, y=y, ax=ax, color=color, alpha=alpha)
                corr = df[x].corr(df[y])
                ax.annotate(f"r = {corr:.2f}", xy=(0.05, 0.85), xycoords='axes fraction',
                            fontsize=9, backgroundcolor='white')
            elif i == j:
                sns.histplot(df[x], ax=ax, color=color, kde=True)
                ax.set_ylabel('')
                ax.set_xlabel('')
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.show()

# Function for scatter plot
# df['column_name', 'column_name', 'column_categorical']
# plot_scatter(df, x_col='sepal_length', y_col='sepal_width', title='Sepal Length vs. Width by Species', xticks_range=range(0, 50, 5), 
#             yticks_range=range(0, 2, 1), hue='species', palette='Set1', alpha=0.6, marker='o', x_rotation=0, y_rotation=0)
def plot_scatter(df, x_col, y_col, title=None, xlabel=None, ylabel=None, figsize=(15, 7), alpha=0.3, color='grey', marker='o',
                 hue=None, palette=None, xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0):
    """
    Plots a scatterplot for two numerical columns with optional customization.

    Parameters:
    - df (DataFrame): Data source.
    - x_col (str): Column name for x-axis.
    - y_col (str): Column name for y-axis.
    - title (str, optional): Plot title.
    - xlabel (str, optional): Label for the x-axis.
    - ylabel (str, optional): Label for the y-axis.
    - figsize (tuple): Figure size (width, height).
    - alpha (float): Transparency level for points.
    - color (str): Color of the points.
    - marker (str): Marker style for scatter points.
    - hue (str, optional): Column name to use for color grouping.
    - palette: color palette if hue is used
    - xticks_range (range, optional): Custom range for x-axis ticks.
    - yticks_range (range, optional): Custom range for y-axis ticks.
    - x_rotation (int): Rotation angle for x-axis tick labels.
    - y_rotation (int): Rotation angle for y-axis tick labels.
    
    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=alpha, color=None if hue else color, palette=palette, marker=marker)

    plt.title(title if title else f'Scatter: {x_col} vs. {y_col}')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)

    if isinstance(xticks_range, range):
        plt.xticks(xticks_range, rotation=x_rotation)
        plt.xlim(xticks_range.start, xticks_range.stop)

    if isinstance(yticks_range, range):
        plt.yticks(yticks_range, rotation=y_rotation)
        plt.ylim(yticks_range.start, yticks_range.stop)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Function to plot ECDF - Empirical Cumulative Distribution Function. Unlike histograms, ECDFs Don’t require selecting bin sizes. 
#                         You can detect outliers by seeing abrupt jumps or flat regions near the start/end of the ECDF.
# ds | df['column_name']
# plot_ecdf(df, x_col='total_bill', threshold=30, title='ECDF of Total Bill with Quartiles and Threshold', xlabel='Total Bill ($)',
#           ylabel='Proportion of Observations', color='steelblue', xticks_range=range(0, 60, 5), yticks_range=range(0, 2, 1), show_quartiles=True)
def plot_ecdf(df, x_col, threshold=None, title=None, xlabel=None, ylabel='Cumulative Percentage', figsize=(10, 6), color='blue', 
              linestyle='--', lw=1.5, xticks_range=None, yticks_range=None, x_rotation=0, y_rotation=0, grid=True,
              show_quartiles=True):
    """
    Plots an Empirical Cumulative Distribution Function (ECDF) for a given column.

    Parameters:
    - df (DataFrame): Data source.
    - x_col (str): Column to plot ECDF for.
    - threshold (float, optional): Draw vertical reference line at this x value.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Figure size.
    - color (str): Color of the ECDF line.
    - linestyle (str): Style for the vertical threshold line.
    - lw (float): Line width for the threshold.
    - xticks_range (range, optional): Custom xtick positions.
    - yticks_range (range, optional): Custom ytick positions.
    - x_rotation (int): Rotation angle for x-axis labels.
    - y_rotation (int): Rotation angle for y-axis labels.
    - grid (bool): Whether to display the grid.
    - show_quartiles (bool): Whether to display vertical reference lines for the first (Q1), second (Q2, median), and third (Q3) quartiles. 
      Useful for visualizing data spread and central tendency.

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.ecdfplot(data=df, x=x_col, color=color)
    
    # Plot quartiles if requested
    if show_quartiles:
        q1 = df[x_col].quantile(0.25)
        q2 = df[x_col].quantile(0.50)
        q3 = df[x_col].quantile(0.75)

        plt.axvline(q1, color='orange', linestyle=':', linewidth=1.5, label=f'Q1 (25%): {q1:.2f}')
        plt.axvline(q2, color='green', linestyle='--', linewidth=1.5, label=f'Median (Q2): {q2:.2f}')
        plt.axvline(q3, color='purple', linestyle=':', linewidth=1.5, label=f'Q3 (75%): {q3:.2f}')

    # Optional threshold line
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle=linestyle, linewidth=lw,
                    label=f'Threshold: {threshold}')

    plt.title(title if title else f'ECDF of {x_col}')
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel)

    if isinstance(xticks_range, range):
        plt.xticks(xticks_range, rotation=x_rotation)
        plt.xlim(xticks_range.start, xticks_range.stop)

    if isinstance(yticks_range, range):
        plt.yticks(yticks_range, rotation=y_rotation)
        plt.ylim(yticks_range.start, yticks_range.stop)

    if threshold is not None:
        plt.legend()

    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# plot_plan_revenue_by_city(df_city_revenue, color1='darkblue', color2='silver', title='Ingresos por ciudad y plan')
# df['column1_name', 'column2_name', 'column3_name']
# plot_bar_comp(df, x_col='City', y_col=['Plan_A', 'Plan_B'], title='Revenue Comparison by City', xlabel='City', ylabel='Revenue ($)',
#               color1='royalblue', color2='lightgrey', rotation=45, fontsize=10)
def plot_bar_comp(df, x_col, y_col, title, xlabel, ylabel, color1='black', color2='grey', alpha2=0.7,
                  figsize=(15, 7), rotation=0, fontsize=8, show_values=True):
    """
    Plots a grouped bar chart comparing revenue by city for two different plans.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the X-axis (e.g., city)
    - plan1: Column name for first plan revenue
    - plan2: Column name for second plan revenue
    - color1: Color for the first plan bars
    - color2: Color for the second plan bars
    - alpha2: Transparency for second plan bars
    - title: Title of the plot
    - xlabel: Label for X-axis
    - ylabel: Label for Y-axis
    - figsize: Size of the figure
    - rotation: Rotation angle for x-axis labels
    - fontsize: Font size for x-axis labels
    - show_values: Whether to display the value above each bar
    """
    plt.figure(figsize=figsize)
    bars1 = plt.bar(df[x_col], df[y_col[0]], label=y_col[0].upper(), color=color1)
    bars2 = plt.bar(df[x_col], df[y_col[1]], label=y_col[1].upper(), color=color2, alpha=alpha2)

    if show_values:
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.xticks(range(len(df[x_col])), df[x_col], rotation=rotation, fontsize=fontsize)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a frequency density histogram with KDE, showing distribution shape and key statistics for sample lower than 5000. 
# Distribution diagnostics: skewness, kurtosis, normality, with Shapiro - Wilk  for test normality
# Highlights mean, median, ±1σ, ±2σ, ±3σ boundaries, and marks outliers beyond 3σ.
# plot_distribution_dispersion(df, 'total_spent', bins='sturges', color='skyblue') 
def plot_distribution_dispersion_sl5000(data, column_name, bins='fd', rug=True, color='grey'):
    """
    Plot a frequency density histogram with KDE, sigma bands, and statistical insights.

    Parameters:
    - data: DataFrame with the column to analyze.
    - column_name: Name of the column to visualize.
    - bins: Number of bins or method name ('sturges', 'sqrt', 'fd', 'rice', 'auto').
    - rug: Whether to display a rugplot (default: True).
    - color: Histogram base color (default: 'grey').

    Displays:
    - Histogram with KDE.
    - Mean, median, ±1σ, ±2σ, ±3σ lines.
    - Outliers beyond 3σ.
    - Distribution diagnostics: skewness, kurtosis, normality.
    """

    values = data[column_name].dropna()

    if isinstance(bins, str):
        method_used = bins
        bins = calculate_bins(values, method=method_used)
        print(f'✅ Using {bins} bins calculated by the method "{method_used}"')

    # Statistics
    mean = values.mean()
    median = values.median()
    std = values.std()
    skewness = values.skew()
    kurtosis = values.kurt()

    # Normality test
    stat, p = shapiro(values)
    is_normal = p >= 0.05

    # Visual styles
    kde_color = 'blue' if is_normal else 'darkred'
    mean_line_style = '--' if is_normal else '-.'
    sigma_colors = ['green', 'blue', 'purple']

    sigma_bounds = {
        '1σ': (mean - std, mean + std),
        '2σ': (mean - 2*std, mean + 2*std),
        '3σ': (mean - 3*std, mean + 3*std)
    }

    # Plot
    plt.figure(figsize=(15, 7))

    # Histogram (no internal KDE)
    sns.histplot(values, bins=bins, kde=False, stat='density',
                 color=color, edgecolor='black', alpha=0.6)

    # manual KDE (color)
    sns.kdeplot(values, color=kde_color, linewidth=2)

    if rug:
        sns.rugplot(values, color='black', alpha=0.15)

    # Mean and median
    plt.axvline(mean, color='red', linestyle=mean_line_style, linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='orange', linestyle=':', linewidth=2, label=f'Median: {median:.2f}')

    # standard deviation lines
    for i, (label, (low, high)) in enumerate(sigma_bounds.items()):
        plt.axvline(low, color=sigma_colors[i], linestyle='--', alpha=0.7, label=f'{label} Lower: {low:.2f}')
        plt.axvline(high, color=sigma_colors[i], linestyle='--', alpha=0.7, label=f'{label} Upper: {high:.2f}')

    # Outliers ±3σ
    outliers = values[(values < sigma_bounds['3σ'][0]) | (values > sigma_bounds['3σ'][1])]
    if not outliers.empty:
        plt.scatter(outliers, np.zeros_like(outliers), color='black', s=40, label='Outliers (3σ+)', marker='x')

    # Title with diagnostics
    dist_label = 'Normal' if is_normal else 'Not Normal'
    plt.title(
        f'Distribution of {column_name} ({dist_label} | Skew={skewness:.2f}, Kurtosis={kurtosis:.2f}, Shapiro p={p:.4f})',
        fontsize=14
    )
    plt.xlabel(column_name)
    plt.ylabel('Frequency Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a frequency density histogram with KDE, showing distribution shape and key statistics for sample greater than 5000. 
# Distribution diagnostics: skewness, kurtosis, normality, with prctic heuristics (skweness, kurtosis) for test normality
# Highlights mean, median, ±1σ, ±2σ, ±3σ boundaries, and marks outliers beyond 3σ.
# plot_distribution_dispersion(df, 'total_spent', bins='sturges', color='skyblue') 
def plot_distribution_dispersion_sg5000(data, column_name, bins='fd', rug=True, color='grey'):
    """
    Plot a frequency density histogram with KDE, sigma bands, and statistical insights.
    Optimized for large datasets (N > 5000), with heuristic detection of normality (no Shapiro).

    Parameters:
    - data: DataFrame with the column to analyze.
    - column_name: Name of the column to visualize.
    - bins: Number of bins or binning method ('sturges', 'sqrt', 'fd', 'rice', 'auto').
    - rug: Whether to display a rugplot (default: False).
    - color: Histogram base color (default: 'grey').

    Displays:
    - Histogram with KDE.
    - Mean, median, ±1σ, ±2σ, ±3σ lines.
    - Outliers beyond 3σ.
    - Title includes skew, kurtosis, sample size and inferred normality.
    """

    values = data[column_name].dropna()

    if isinstance(bins, str):
        method_used = bins
        bins = calculate_bins(values, method=method_used)
        print(f'✅ Usando {bins} bins calculados con el método "{method_used}"')

    # Estadísticas
    mean = values.mean()
    median = values.median()
    std = values.std()
    skewness = values.skew()
    kurtosis = values.kurt()

    # Heurística de normalidad
    is_normal_like = abs(skewness) < 0.5 and abs(kurtosis) < 1

    kde_color = 'darkblue' if is_normal_like else 'darkred'
    mean_line_style = '--' if is_normal_like else '-.'
    sigma_colors = ['green', 'blue', 'purple']
    normality_label = "Normal" if is_normal_like else "Not Normal"

    sigma_bounds = {
        '1σ': (mean - std, mean + std),
        '2σ': (mean - 2*std, mean + 2*std),
        '3σ': (mean - 3*std, mean + 3*std)
    }

    # Plot
    plt.figure(figsize=(15, 7))

    # Histograma
    sns.histplot(values, bins=bins, kde=False, stat='density',
                 color=color, edgecolor='black', alpha=0.6)

    # KDE
    sns.kdeplot(values, color=kde_color, linewidth=2)

    # Rugplot (opcional y limitado)
    if rug and len(values) <= 50000:
        sns.rugplot(values, color='black', alpha=0.15)

    # Líneas de media y mediana
    plt.axvline(mean, color='red', linestyle=mean_line_style, linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='orange', linestyle=':', linewidth=2, label=f'Median: {median:.2f}')

    # Líneas sigma
    for i, (label, (low, high)) in enumerate(sigma_bounds.items()):
        plt.axvline(low, color=sigma_colors[i], linestyle='--', alpha=0.7, label=f'{label} Lower: {low:.2f}')
        plt.axvline(high, color=sigma_colors[i], linestyle='--', alpha=0.7, label=f'{label} Upper: {high:.2f}')

    # Outliers
    outliers = values[(values < sigma_bounds['3σ'][0]) | (values > sigma_bounds['3σ'][1])]
    if not outliers.empty:
        plt.scatter(outliers, np.zeros_like(outliers), color='black', s=40, label='Outliers (3σ+)', marker='x')

    # Título final
    plt.title(
        f'Distribution of {column_name} ({normality_label} | Skew={skewness:.2f}, Kurtosis={kurtosis:.2f}, N={len(values)})',
        fontsize=14
    )
    plt.xlabel(column_name)
    plt.ylabel('Frequency Density')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plots a vertical bar chart from a pandas Series with customizable labels, size, and ticks.
# avg_tip_series = tips.groupby('day')['tip'].mean().round(2)
# ds | df['column_name']
# custom_xticks = ['Thur', 'Fri', 'Sat', 'Sun']
# custom_yticks = list(range(0, 5, 1))
# plot_bar_series(ds, title='Average Tip Amount by Day', xlabel='Day of Week', ylabel='Average Tip ($)', color='mediumseagreen',
#                 rotation=0, show_values=True, xticks=custom_xticks, yticks=custom_yticks)
def plot_bar_series(series, title='', xlabel='', ylabel='', figsize=(15, 7), color=None, rotation=45, show_values=True, xticks=None, 
                    yticks=None):
    """
    Creates a vertical bar chart from a Pandas Series.

    Parameters:
    - series: pd.Series — Series with indices as categories and numeric values.
    - title: str — Chart title.
    - xlabel: str — X-axis label.
    - ylabel: str — Y-axis label.
    - figsize: tuple — Figure size (width, height).
    - color: str or list — Color(s) of the bars.
    - rotation: int — Rotation of the labels on the X-axis.
    - show_values: bool — Show values ​​above the bars.
    - xticks: list or None — Custom list of values ​​for the X-axis.
    - yticks: list or None — Custom list of values ​​for the Y-axis.

    Returns:
    - None
    """

    fig, ax = plt.subplots(figsize=figsize)

    series.plot(kind='bar', ax=ax, color=color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=rotation)

    if xticks is not None:
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    # Show values above bars
    if show_values:
        for i, val in enumerate(series):
            ax.text(i, val + max(series)*0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plots horizontal lines.
# df['column1_name', 'columns2_name', 'column3_name']
# plot_horizontal_lines(df, start_col='Revenue_2023', end_col='Revenue_2024', y_col='City', title='Revenue Growth: 2023 to 2024', xlabel='Revenue ($)',
#                       ylabel='City', marker='o', color='darkslategray')
def plot_horizontal_lines(df, start_col='', end_col='', y_col='', title='', xlabel='', ylabel='', figsize=(15, 7), marker='o', grid=True, color='tab:grey'):
    """
    Plots horizontal lines.

    Parameters:
    - df: DataFrame containing data.
    - start_col: column name with the first column values.
    - end_col: column name with the last column values.
    - y_col: column name with the oject values.
    - title: title of the chart.
    - xlabel: label for the X axis.
    - ylabel: label for the Y axis.
    - figsize: figure size (width, height).
    - marker: marker style for endpoints (default is 'o').
    - grid: whether to display a grid (default is True).
    - color: line color (string or list of colors for each platform).

    Returns:
    - fig, ax: matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, row in df.iterrows():
        line_color = color[i] if isinstance(color, list) else color
        ax.plot([row[start_col], row[end_col]],
                [row[y_col], row[y_col]],
                marker=marker,
                color=line_color)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Plots Quantile to Quantile graph
# plot_qq_normality_tests(df, 'column_name')
def plot_qq_normality_tests(data, column, dist='norm', dist_params=None,
                                  line='45', title=None, figsize=(15, 7), color='grey',
                                  outlier_color='black', outlier_marker='x'):
    """
    Generate a QQ plot comparing the quantiles of a sample against a theoretical distribution.
    Outliers are detected using the IQR method and highlighted with custom color and marker.
    Also performs normality tests and displays the results.

    Parameters:
    - data: pd.DataFrame – The dataset containing the column to analyze.
    - column: str – The column name to analyze.
    - dist: str or scipy.stats distribution – The distribution to compare against (default: 'norm').
    - dist_params: tuple – Parameters required for the theoretical distribution (shape, loc, scale).
    - line: str – Reference line type for QQ plot.
    - title: str – Plot title.
    - figsize: tuple – Size of the figure.
    - color: str – Color of the main data points.
    - outlier_color: str – Color of the outlier points.
    - outlier_marker: str – Marker style for outliers.
    
    """

    # Drop NA values
    values = data[column].dropna().values
    n = len(values)

    # Determine theoretical distribution
    dist_obj = getattr(stats, dist) if isinstance(dist, str) else dist

    # Generate QQ plot data
    if dist_params is not None:
        (osm, osr), (slope, intercept, r) = stats.probplot(values, dist=dist_obj, sparams=dist_params)
    else:
        (osm, osr), (slope, intercept, r) = stats.probplot(values, dist=dist_obj)

    # Detect outliers using IQR method on sample quantiles
    Q1 = np.percentile(osr, 25)
    Q3 = np.percentile(osr, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = (osr < lower_bound) | (osr > upper_bound)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(osm[~outlier_mask], osr[~outlier_mask], color=color, label='Data', marker='o')
    ax.scatter(osm[outlier_mask], osr[outlier_mask], color=outlier_color, label='IQR Outliers', marker=outlier_marker)
    ax.plot(osm, slope * osm + intercept, 'r-', lw=2, label=f'{dist_obj.name.title()} Fit')

    ax.set_title(title or f'QQ Plot ({column}) vs. {dist_obj.name.title()}')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.legend()
    ax.grid(True)

    # Perform all 3 normality tests
    shapiro_stat, shapiro_p = stats.shapiro(values)
    dagostino_stat, dagostino_p = stats.normaltest(values)
    ad_result = stats.anderson(values, dist='norm')
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values[2]  # 5% level

    # Build results table
    html = f"""
    <h4>Normality Tests for <code>{column}</code> (n={n})</h4>
    <table border="1" style="border-collapse:collapse; text-align:center;">
    <tr>
        <th>Test</th>
        <th>Statistic</th>
        <th>p-value / Critical</th>
        <th>Conclusion</th>
        <th>Recommended for</th>
        <th>Sensitive to</th>
    </tr>
    """

    conclusion_sw = "Reject H₀ (Not Normal)" if shapiro_p < 0.05 else "Fail to Reject H₀ (Possibly Normal)"
    html += f"<tr><td>Shapiro-Wilk</td><td>{shapiro_stat:.4f}</td><td>{shapiro_p:.4f}</td><td>{conclusion_sw}</td><td>n ≤ 5000</td><td>General deviations</td></tr>"

    conclusion_dp = "Reject H₀ (Not Normal)" if dagostino_p < 0.05 else "Fail to Reject H₀ (Possibly Normal)"
    html += f"<tr><td>D’Agostino-Pearson</td><td>{dagostino_stat:.4f}</td><td>{dagostino_p:.4f}</td><td>{conclusion_dp}</td><td>n > 500</td><td>Skewness & Kurtosis</td></tr>"

    conclusion_ad = "Reject H₀ (Not Normal)" if ad_stat > ad_crit else "Fail to Reject H₀ (Possibly Normal)"
    html += f"<tr><td>Anderson-Darling</td><td>{ad_stat:.4f}</td><td>Crit: {ad_crit:.4f}</td><td>{conclusion_ad}</td><td>All sample sizes</td><td>Tail behavior</td></tr>"

    html += "</table>"

    display(HTML(html))

# Plot horizontal boxplot
# plot_horizontal_boxplot(df, 'column_name')
def plot_horizontal_boxplot(data, column, figsize=(15, 5), box_color='lightgrey',
                       point_color='black', outlier_color='red',
                       point_marker='.', outlier_marker='x'):
    """
    Horizontal boxplot with aligned markers:
    - Points, outliers and mean marker are aligned with whiskers (y=1).
    - Red diamond shows mean.
    - Annotates mean, median, and IQR-based outlier thresholds.

    Parameters:
    - data: pd.DataFrame – DataFrame containing the column to plot.
    - column: str – Column name to visualize.
    - figsize: tuple – Size of the figure (width, height).
    - box_color: str – Color of the boxplot.
    - point_color: str – Color of non-outlier data points.
    - outlier_color: str – Color of outlier points.
    - point_marker: str – Marker style for regular data points.
    - outlier_marker: str – Marker style for outliers.

    """

    values = data[column].dropna().values

    # Compute statistics
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_val = np.mean(values)
    median_val = np.median(values)

    # Identify outliers
    outliers = values[(values < lower_bound) | (values > upper_bound)]
    non_outliers = values[(values >= lower_bound) & (values <= upper_bound)]

    # Initialize plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(values, vert=False, patch_artist=True,
               boxprops=dict(facecolor=box_color, color='black'),
               medianprops=dict(color='blue'),
               whiskerprops=dict(color='red'),
               capprops=dict(color='black'),
               flierprops=dict(marker='', linestyle='none'))  # custom outliers only

    # Plot all data at y=1 (aligned with boxplot)
    ax.scatter(non_outliers, np.full_like(non_outliers, 1), color=point_color,
               marker=point_marker, label='Data Points', zorder=3)
    ax.scatter(outliers, np.full_like(outliers, 1), color=outlier_color,
               marker=outlier_marker, label='IQR Outliers', zorder=4)
    ax.scatter(mean_val, 1, color='red', marker='D', s=70, label='Mean', zorder=5)

    # Annotations
    ax.annotate(f"Mean = {mean_val:.2f}", xy=(mean_val, 1), xytext=(0, 12),
                textcoords='offset points', ha='center', color='red', fontsize=7, weight='bold')
    ax.annotate(f"Median = {median_val:.2f}", xy=(median_val, 1), xytext=(0, 20),
                textcoords='offset points', ha='center', color='blue', fontsize=7)
    ax.annotate(f"← Outlier threshold = {lower_bound:.2f}", xy=(lower_bound, 1), xytext=(-5, -15),
                textcoords='offset points', ha='right', color='crimson', fontsize=7)
    ax.annotate(f"Outlier threshold = {upper_bound:.2f} →", xy=(upper_bound, 1), xytext=(5, -15),
                textcoords='offset points', ha='left', color='crimson', fontsize=7)

    # Styling
    ax.set_yticks([])
    ax.set_ylim(0.9, 1.1)  # Keep everything centered around y=1
    ax.set_title(f'Boxplot with Stats: {column}', fontsize=11, weight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend()

# ---

# Theoretical distributions and their parameters in scipy.stats for QQ plot

# | Distribution (`scipy.stats`) | Required `dist_params`  | Parameter meaning                                    | When to use it                                                               |
# | ---------------------------- | ----------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
# | `norm`                       | ❌ None                  | Defaults: `loc=0`, `scale=1`                         | When data appears **symmetric and bell-shaped** (normal)                     |
# | `lognorm`                    | ✅ `(shape, loc, scale)` | `shape` = log-σ, `loc` = shift, `scale` = median-ish | For **right-skewed**, strictly **positive** data (e.g., income, price, time) |
# | `expon`                      | ✅ `(loc, scale)`        | `loc` = start point, `scale` = mean                  | For **time between events**, **rapid decay** (e.g., Poisson processes)       |
# | `gamma`                      | ✅ `(a, loc, scale)`     | `a` = shape parameter, controls skewness             | For **right-skewed data**, but more flexible than log-normal                 |
# | `beta`                       | ✅ `(a, b, loc, scale)`  | `a`, `b` = shape params, `loc/scale` = range (0–1)   | For **bounded data**, such as **proportions** or percentages                 |
# | `chi2`                       | ✅ `(df, loc, scale)`    | `df` = degrees of freedom                            | For **variance**, squared errors, or **sum of squares**                      |
# | `weibull_min`                | ✅ `(c, loc, scale)`     | `c` = shape parameter                                | For **lifetimes**, **failure analysis**, survival modeling                   |
# | `uniform`                    | ✅ `(loc, scale)`        | `loc` = minimum, `scale` = range                     | For data **evenly distributed** across a range (rare in real-world data)     |
# | `t` (Student's t)            | ✅ `(df, loc, scale)`    | `df` = degrees of freedom                            | Like normal, but **fatter tails** — useful for **small samples**             |

# ---

# Best practices for choosing "cmaps"
# - Sequential (e.g. viridis, plasma): best for ordered data.
# - Diverging (RdBu, coolwarm): when data deviates around a central midpoint.
# - Qualitative (Set1, Paired): to color discrete categories.
# - Cyclic (twilight): for datasets that wrap around, like wind direction or phase angles.

# - Perceptually Uniform Sequential
#   viridis, plasma, inferno, magma, cividis

# - Other Sequential
#   Greys, Purples, Blues, Greens, Oranges, Reds, YlOrBr, YlOrRd, OrRd, PuRd, RdPu, BuPu, GnBu, PuBu, YlGnBu, PuBuGn, BuGn, YlGn

# - Sequential (2)
#   binary, gist_yarg, gist_gray, gray, bone, pink, spring, summer, autumn, winter, cool, Wistia, hot, afmhot, gist_heat, copper

# - Diverging
#   PiYG, PRGn, BrBG, PuOr, RdGy, RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic, berlin, managua, vanimo

# - Cyclic
#   twilight, twilight_shifted, hsv

# - Qualitative
#   Pastel1, Pastel2, Paired, Accent, Dark2, Set1, Set2, Set3, tab10, tab20, tab20b, tab20c

# - Miscellaneous / Special-use
#   flag, prism, ocean, gist_earth, terrain, gist_stern, gnuplot, gnuplot2, CMRmap, cubehelix, brg, gist_rainbow, rainbow, jet, turbo, nipy_spectral, gist_ncar

# ---

# Distribution	            Suggested Method
# ------------              ----------------
# Normal                	Sturges, Ideal for approximately normal distributions.
#                           Scott, 
#                           Rice, when there is a lot of data and you want more stable bins, Large n, smooth distributions
# Samll n             	    Square Root, Simple and quick. A good starting point.
# with outliers	            Freedman–Diaconis, Excellent for data with outliers. Uses the interquartile range (IQR).
# No idea	                bins='auto'



# ---
# Plotly Express plotting functions
# ---

# Function to visualize missing values within a DataFrame using a heatmap
# missing_values_heatmap_plotlypx(df)
def missing_values_heatmap_plotlypx(df, title='Heatmap of Missing Values'):
    """
    Interactive heatmap of missing values using Plotly Express.
    
    Parameters:
    - df (DataFrame): Input DataFrame.
    - title (str): Title of the heatmap.
    
    Output:
    Displays an interactive heatmap.
    """
    missing_data = df.isna().astype(int)
    fig = px.imshow(
        missing_data,
        labels=dict(x="Columns", y="Rows", color="Missing"),
        color_continuous_scale=[[0, 'darkred'], [1, 'yellow']],
        zmin=0, zmax=1,
        aspect='auto',
        title=title
    )
    fig.update_layout(
        yaxis=dict(visible=False),
        xaxis_tickangle=0,
        xaxis=dict(tickfont=dict(size=10))  # Tamaño de letra más pequeño
    )
    fig.show()


# Plots Quantile to Quantile graph
# plot_qq_normality_tests_plotlypx(df, 'column_name')
def plot_qq_normality_tests_plotlypx(data, column, dist='norm', dist_params=None,
                                    title=None, color='grey', outlier_color='crimson',
                                    outlier_marker='x', width=1200, height=600):
    """
    Interactive QQ plot using Plotly, including normality tests and outlier detection.
    
    Generate a QQ plot comparing the quantiles of a sample against a theoretical distribution.
    Outliers are detected using the IQR method and highlighted with custom color and marker.
    Also performs normality tests and displays the results.

    Parameters:
    - data: pd.DataFrame – The dataset containing the column to analyze.
    - column: str – The column name to analyze.
    - dist: str or scipy.stats distribution – The distribution to compare against (default: 'norm').
    - dist_params: tuple – Parameters required for the theoretical distribution (shape, loc, scale).
    - title: str – Plot title.
    - color: str – Color of the main data points.
    - outlier_color: str – Color of the outlier points.
    - outlier_marker: str – Marker style for outliers.
    """
    values = data[column].dropna().values
    n = len(values)

    # Get theoretical distribution
    dist_obj = getattr(stats, dist) if isinstance(dist, str) else dist

    # QQ data
    if dist_params:
        (osm, osr), (slope, intercept, r) = stats.probplot(values, dist=dist_obj, sparams=dist_params)
    else:
        (osm, osr), (slope, intercept, r) = stats.probplot(values, dist=dist_obj)

    # Outlier detection via IQR on sample quantiles
    Q1, Q3 = np.percentile(osr, [25, 75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    is_outlier = (osr < lower) | (osr > upper)

    # Plotly figure
    fig = go.Figure()

    # Non-outliers
    fig.add_trace(go.Scatter(
        x=osm[~is_outlier], y=osr[~is_outlier],
        mode='markers',
        name='Data',
        marker=dict(color=color, size=6),
        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))

    # Outliers
    fig.add_trace(go.Scatter(
        x=osm[is_outlier], y=osr[is_outlier],
        mode='markers',
        name='IQR Outliers',
        marker=dict(color=outlier_color, size=8, symbol=outlier_marker),
        hovertemplate='Outlier<br>Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))

    # Reference line
    line_x = np.array([osm.min(), osm.max()])
    line_y = slope * line_x + intercept
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode='lines',
        name=f'{dist_obj.name.title()} Fit',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=title or f'QQ Plot ({column}) vs. {dist_obj.name.title()}',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
        width=width,
        height=height,
        template='simple_white'
    )

    fig.show()

    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(values)
    dagostino_stat, dagostino_p = stats.normaltest(values)
    ad_result = stats.anderson(values, dist='norm')
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values[2]

    # Summary table
    html = f"""
    <h4>Normality Tests for <code>{column}</code> (n={n})</h4>
    <table border="1" style="border-collapse:collapse; text-align:center;">
    <tr><th>Test</th><th>Statistic</th><th>p-value / Critical</th><th>Conclusion</th><th>Recommended for</th><th>Sensitive to</th></tr>
    <tr><td>Shapiro-Wilk</td><td>{shapiro_stat:.4f}</td><td>{shapiro_p:.4f}</td><td>{'Reject H₀ (Not Normal)' if shapiro_p < 0.05 else 'Possibly Normal'}</td><td>n ≤ 5000</td><td>General deviations</td></tr>
    <tr><td>D’Agostino-Pearson</td><td>{dagostino_stat:.4f}</td><td>{dagostino_p:.4f}</td><td>{'Reject H₀ (Not Normal)' if dagostino_p < 0.05 else 'Possibly Normal'}</td><td>n > 500</td><td>Skewness & Kurtosis</td></tr>
    <tr><td>Anderson-Darling</td><td>{ad_stat:.4f}</td><td>Crit: {ad_crit:.4f}</td><td>{'Reject H₀ (Not Normal)' if ad_stat > ad_crit else 'Possibly Normal'}</td><td>All sizes</td><td>Tail behavior</td></tr>
    </table>
    """
    display(HTML(html))



# Plot horizontal boxplot
# plot_horizontal_boxplotpx(df, 'column_name')
def plot_horizontal_boxplot_plotlypx(data, column, title=None):
    """
    Horizontal boxplot with aligned markers:
    - Points, outliers and mean marker are aligned with whiskers (y=1).
    - Red diamond shows mean.
    - Annotates mean, median, and IQR-based outlier thresholds.

    Parameters:
    - data: pd.DataFrame – DataFrame containing the column to plot.
    - column: str – Column name to visualize.
    - title: str - Title to visualize
    """
    values = data[column].dropna().values
    Q1, Q3 = np.percentile(values, [25, 75])
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mean_val = np.mean(values)
    median_val = np.median(values)

    outliers = values[(values < lower_bound) | (values > upper_bound)]
    non_outliers = values[(values >= lower_bound) & (values <= upper_bound)]

    fig = go.Figure()

    # 1. Boxplot principal (solo datos sin outliers)
    fig.add_trace(go.Box(
        x=non_outliers,
        y=['Data'] * len(non_outliers),
        name='Boxplot',
        orientation='h',
        boxpoints=False,
        fillcolor='lightgrey',
        line=dict(color='black'),
        showlegend=False
    ))

    # 2. Puntos normales
    fig.add_trace(go.Scatter(
        x=non_outliers,
        y=['Data'] * len(non_outliers),
        mode='markers',
        name='Data Points',
        marker=dict(color='black', symbol='circle', size=5),
        hoverinfo='x'
    ))

    # 3. Outliers
    fig.add_trace(go.Scatter(
        x=outliers,
        y=['Data'] * len(outliers),
        mode='markers',
        name='IQR Outliers',
        marker=dict(color='red', symbol='x', size=8),
        hoverinfo='x'
    ))

    # 4. Media (rombo rojo)
    fig.add_trace(go.Scatter(
        x=[mean_val],
        y=['Data'],
        mode='markers+text',
        name='Mean',
        marker=dict(color='red', symbol='diamond', size=10),
        text=[f"Mean = {mean_val:.2f}"],
        textposition='top center',
        textfont=dict(color='red')
    ))

    # 5. Mediana (rombo azul)
    fig.add_trace(go.Scatter(
        x=[median_val],
        y=['Data'],
        mode='markers+text',
        name='Median',
        marker=dict(color='blue', symbol='diamond', size=10),
        text=[f"Median = {median_val:.2f}"],
        textposition='bottom center',
        textfont=dict(color='blue')
    ))

    # 6. Líneas verticales para límites del IQR
    fig.add_shape(type="line",
                  x0=lower_bound, y0=0.9,
                  x1=lower_bound, y1=1.1,
                  line=dict(color="red", dash="dot"))

    fig.add_shape(type="line",
                  x0=upper_bound, y0=0.9,
                  x1=upper_bound, y1=1.1,
                  line=dict(color="red", dash="dot"))

    # Layout
    fig.update_layout(
        title=title or f'Boxplot with Stats: {column}',
        xaxis_title=column,
        yaxis=dict(showticklabels=False),
        height=450,
        template='simple_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.show()



# Function to plot a frequency density histogram with optional KDE overlay
# ds | df['column_name']
# plot_frequency_density_plotlypx(tips['tip'], bins=20, color='grey', title='Density Plot for Tips', xlabel='Tip Amount ($)', ylabel='Density',
#                        xticks_range=(0, 11, 1), rotation=45, show_kde=True)
def plotly_frequency_density_plotlypx(ds, bins=10, density=True, color='grey', title='', xlabel='', ylabel='Frequency/Density',
                                      xticks_range=None, rotation=0, show_kde=True):
    """
    Interactive frequency density histogram with optional KDE overlay using Plotly.

    Parameters:
    - ds (Series): Numerical data to plot.
    - bins (int or array-like): Number or range of bins for the histogram.
    - density(True): True for Probability density and False for Frequency
    - color (str): Histogram bar color.
    - title (str): Plot title.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - xticks_range (tuple, optional): Tuple (min, max, step) for x-tick configuration.
    - rotation (int, optional): Angle for tick label rotation (not supported in Plotly).
    - show_kde (bool, optional): Whether to overlay a KDE curve.

    Output:
    Displays an interactive histogram with density normalization, mean/median lines, and optional KDE.
    """
    ds = ds.dropna()
    mean_val = ds.mean()
    median_val = ds.median()

    # Histogram data
    hist_data = np.histogram(ds, bins=bins, density=density)
    bin_edges = hist_data[1]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    densities = hist_data[0]
    frequencies = hist_data[0]

    fig = go.Figure()

    # Add histogram bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=densities if density == True else frequencies,
        marker_color=color,
        opacity=0.7,
        name='Histogram',
        width=np.diff(bin_edges),
    ))

    # Add KDE curve
    if show_kde:
        kde = gaussian_kde(ds)
        x_vals = np.linspace(ds.min(), ds.max(), 500)
        kde_vals = kde(x_vals) if density == True else (kde(x_vals) * len(ds) * np.diff(bin_edges)[0])
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=kde_vals,
            mode='lines',
            line=dict(color='darkblue', width=2),
            name='KDE'
        ))

    # Add mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_val, mean_val],
        y=[0, max(densities)*1.1] if density == True else [0, max(frequencies)*1.1],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f'Mean: {mean_val:.2f}'
    ))

    fig.add_trace(go.Scatter(
        x=[median_val, median_val],
        y=[0, max(densities)*1.1] if density == True else [0, max(frequencies)*1.1],
        mode='lines',
        line=dict(color='blue', dash='dashdot'),
        name=f'Median: {median_val:.2f}'
    ))

    # Layout settings
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        bargap=0.05,
        template='plotly_white',
        legend=dict(orientation='v', x=1.02, y=1, xanchor='right', yanchor='top'),
        width=1200,
        height=600
    )

    if xticks_range:
        fig.update_xaxes(
            range=[xticks_range[0], xticks_range[1]],
            tickmode='array',
            tickvals=np.arange(*xticks_range)
        )

    fig.show()

# Function to plot a scatter matrix for exploring pairwise relationships (CORR)
# df | df['column1_name', 'column2_name', ..., 'columnN_name']
# plot_scatter_matrix(df, figsize=(12, 10), diagonal='kde', color='teal', alpha=0.4)

def plot_scatter_matrixpx(df: pd.DataFrame, columns=None, height: int = 600, marker_color: str = "grey", marker_opacity: float = 0.6,
                          bins: int = 20, horizontal_spacing: float = 0.03, vertical_spacing: float = 0.03,):
    """
    Interactive correlation matrix (lower triangle) with:
    - Scatter in cells i>j
    - Histogram + KDE on the diagonal (i=j)
    - Pearson correlation annotation (r) in each scatter cell
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    if len(columns) < 2:
        raise ValueError("At least 2 numeric columns are required.")

    data = df[columns].copy()
    n = len(columns)
    fig = make_subplots(
        rows=n,
        cols=n,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing,
    )

    def axis_suffix(row, col):
        idx = (row - 1) * n + col
        return "" if idx == 1 else str(idx)

    for i, ycol in enumerate(columns, start=1):
        for j, xcol in enumerate(columns, start=1):

            if i > j:  # Lower triangle: scatter + r
                pair = data[[xcol, ycol]].dropna()
                if not pair.empty:
                    fig.add_trace(
                        go.Scattergl(
                            x=pair[xcol],
                            y=pair[ycol],
                            mode="markers",
                            marker=dict(size=5, opacity=marker_opacity, color=marker_color),
                            hovertemplate=f"{xcol}: %{{x}}<br>{ycol}: %{{y}}<extra></extra>",
                            showlegend=False,
                        ),
                        row=i, col=j
                    )
                    r = pair[xcol].corr(pair[ycol])
                    suf = axis_suffix(i, j)
                    fig.add_annotation(
                        x=0.04, y=0.9,
                        xref=f"x{suf} domain", yref=f"y{suf} domain",
                        text=f"r = {r:.2f}",
                        showarrow=False,
                        font=dict(size=12),
                        bgcolor="white",
                        bordercolor="#e0e0e0",
                        borderwidth=1,
                        opacity=0.9,
                    )

                if i == n:
                    fig.update_xaxes(title_text=xcol, row=i, col=j)
                if j == 1:
                    fig.update_yaxes(title_text=ycol, row=i, col=j)

            elif i == j:  # Diagonal: histogram + KDE
                series = data[xcol].dropna()
                if not series.empty:
                    #1) Explicitly calculate bins to find the exact bin_width
                    counts, edges = np.histogram(series, bins=bins)
                    bin_width = edges[1] - edges[0]

                    # 2) Histogram using the same edges (matches 1:1 with np.histogram)
                    fig.add_trace(
                        go.Histogram(
                            x=series,
                            xbins=dict(start=edges[0], end=edges[-1], size=bin_width),
                            marker=dict(color=marker_color),
                            opacity=0.6,
                            showlegend=False,
                            hovertemplate=f"{xcol}: %{{x}}<br>count: %{{y}}<extra></extra>",
                        ),
                        row=i, col=j
                    )

                    # 3) KDE on the same mesh and scaled to "counts"
                    kde = gaussian_kde(series)
                    x_grid = np.linspace(edges[0], edges[-1], 400)  # dense mesh within edges
                    y_kde = kde(x_grid)

                    # Scale to counts: ∫KDE = 1 => height in "counts" ≈ N * bin_width * KDE(x)
                    y_kde_counts = y_kde * len(series) * bin_width

                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=y_kde_counts,
                            mode="lines",
                            line=dict(width=3),   # we don't set color so it inherits from the theme if you want
                            name="KDE",
                            showlegend=False,
                            hovertemplate=f"{xcol}: %{{x}}<br>KDE(counts): %{{y}}<extra></extra>",
                        ),
                        row=i, col=j
                    )

                if i == n:
                    fig.update_xaxes(title_text=xcol, row=i, col=j)
                if j == 1:
                    fig.update_yaxes(title_text="count", row=i, col=j)

            else:
                fig.update_xaxes(visible=False, row=i, col=j)
                fig.update_yaxes(visible=False, row=i, col=j)

    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=40, b=40),
        template="plotly_white",
        title=dict(text="Correlation matrix (lower triangle) with Pearson's r and KDE", x=0.5),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    fig.show()