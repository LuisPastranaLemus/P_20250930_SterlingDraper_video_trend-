# __init__.py is a special file Python looks for when treating a folder as a package.
# Smart __init__.py used for importing easily from the package.

try:

    from .data_loader import (load_dataset_from_zip,
                              load_dataset_from_csv,
                              load_dataset_from_excel,
                              load_dataset_from_list,
                              load_dataset_from_dict)

    from .data_cleaning import (check_existing_missing_values,
                                replace_missing_values,
                                missing_values_rate,
                                normalize_string_format,
                                normalize_columns_headers_format,
                                detect_implicit_duplicates_token,
                                detect_implicit_duplicates_fuzzy,
                                normalize_datetime,
                                find_fail_conversion_to_numeric,
                                convert_object_to_numeric,
                                convert_integer_to_boolean,
                                standardize_gender_values,
                                convert_numday_strday)

    from .eda import (outlier_limit_bounds,
                      evaluate_central_trend,
                      calculate_bins, 
                      evaluate_correlation,
                      missing_values_heatmap,
                      plot_heatmap,
                      plot_boxplots,
                      plot_histogram,
                      plot_hue_histogram,
                      plot_dual_histogram,
                      plot_frequency_density,
                      plot_hue_barplot,
                      plot_categorical_horizontal_bar,
                      plot_horizontal_bar,
                      plot_grouped_bars,
                      plot_pairplot,
                      plot_scatter_matrix,
                      plot_scatter,
                      plot_ecdf,
                      plot_bar_comp,
                      plot_distribution_dispersion_sl5000,
                      plot_distribution_dispersion_sg5000,
                      plot_bar_series,
                      plot_horizontal_lines,
                      plot_qq_normality_tests,
                      plot_horizontal_boxplot)
    
    from .features import(cast_datatypes)

    from .utils import (format_notebook)


except ImportError as e:
    raise ImportError("One or more modules could not be found."
                      "Ensure required scripts exist in the same directory as '__init__.py'.") from e

__all__ = ['load_dataset_from_zip',
           'load_dataset_from_csv',
           'load_dataset_from_excel',
           'load_dataset_from_list',
           'load_dataset_from_dict',

           'check_existing_missing_values',
           'replace_missing_values',
           'missing_values_rate',
           'normalize_string_format',
           'normalize_columns_headers_format',
           'detect_implicit_duplicates_token',
           'detect_implicit_duplicates_fuzzy',
           'normalize_datetime',
           'find_fail_conversion_to_numeric',
           'convert_object_to_numeric',
           'convert_integer_to_boolean',
           'standardize_gender_values',
           'convert_numday_strday',

           'outlier_limit_bounds',
           'evaluate_central_trend',
           'calculate_bins',
           'evaluate_correlation',
           'missing_values_heatmap',
           'plot_heatmap',
           'plot_boxplots',
           'plot_histogram',
           'plot_hue_histogram',
           'plot_dual_histogram',
           'plot_frequency_density',
           'plot_hue_barplot',
           'plot_categorical_horizontal_bar',
           'plot_horizontal_bar',
           'plot_grouped_bars',
           'plot_pairplot',
           'plot_scatter_matrix',
           'plot_scatter',
           'plot_ecdf',
           'plot_bar_comp',
           'plot_distribution_dispersion_sl5000',
           'plot_distribution_dispersion_sg5000',
           'plot_bar_series',
           'plot_horizontal_lines',
           'plot_qq_normality_tests',
           'plot_horizontal_boxplot',
           
           'cast_datatypes',

           'format_notebook']
