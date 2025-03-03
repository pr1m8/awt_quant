.. _multi_factor_analysis:

=================================
Multi-Factor Analysis (MFA) Module
=================================

The **Multi-Factor Analysis (MFA) module** provides a structured framework for financial factor modeling,
machine learning-based feature selection, clustering, and risk analysis. This module is designed to help
analyze portfolios, discover relationships between assets, and perform robust stress testing.

-------------------------
Key Features
-------------------------
- **Data Collection**: Fetches historical price and return data for assets.
- **Factor Construction**: Computes financial indicators used in risk and return modeling.
- **Feature Selection**: Uses machine learning to identify the most relevant financial factors.
- **Clustering**: Groups assets based on similar factor characteristics.
- **Localized Modeling**: Builds specialized models per cluster to improve prediction accuracy.
- **Stress & Sensitivity Analysis**: Evaluates the resilience of asset groups under extreme conditions.

-------------------------
Modules
-------------------------

.. autoclass:: DataCollector
   :members:
   :undoc-members:
   :show-inheritance:

The `DataCollector` class fetches historical price and return data for a set of assets over a given period.

.. autoclass:: FactorConstructor
   :members:
   :undoc-members:
   :show-inheritance:

The `FactorConstructor` class generates various financial indicators and factors for analysis.

.. autoclass:: RandomForestFeatureSelector
   :members:
   :undoc-members:
   :show-inheritance:

The `RandomForestFeatureSelector` applies machine learning to determine the most significant factors in a dataset.

.. autoclass:: KMeansClusterer
   :members:
   :undoc-members:
   :show-inheritance:

The `KMeansClusterer` groups assets based on similar factor characteristics using k-means clustering.

.. autoclass:: LocalizedModel
   :members:
   :undoc-members:
   :show-inheritance:

The `LocalizedModel` trains models separately for each cluster, allowing for more tailored forecasts.

.. autoclass:: StressSensitivityAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

The `StressSensitivityAnalysis` module runs stress tests and sensitivity analysis to assess portfolio resilience.

-------------------------
Example Usage
-------------------------

.. code-block:: python

    from awt_quant.portfolio.multi_factor_analysis import (
        DataCollector, FactorConstructor, RandomForestFeatureSelector,
        KMeansClusterer, LocalizedModel, StressSensitivityAnalysis
    )

    start_date = "2020-01-01"
    end_date = "2022-01-01"
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]

    # Collect financial data
    data_collector = DataCollector(symbols, start_date, end_date)
    price_data = data_collector.fetch_price_data()
    returns_data = data_collector.calculate_historical_returns(price_data)

    # Construct factors
    factor_constructor = FactorConstructor(start_date, end_date)
    factors_df = factor_constructor.get_all_factors()

    # Select important features
    feature_selector = RandomForestFeatureSelector(factor_constructor, data_collector)
    important_features = feature_selector.select_important_features(factors_df)

    # Perform clustering
    kmeans_clusterer = KMeansClusterer(factors_df[important_features])
    clustered_data = kmeans_clusterer.perform_clustering()
    kmeans_clusterer.visualize_clusters(clustered_data)

    # Train localized models
    returns_df = {ticker: df['Daily_Return'] for ticker, df in returns_data.items()}
    localized_model = LocalizedModel(clustered_data, returns_df)
    localized_model.train_all_clusters()

    # Run stress testing
    stress_analysis = StressSensitivityAnalysis(localized_model, clustered_data)
    results = stress_analysis.global_downturn(cluster_id=0, downturn_pct=0.1)
    print("Global downturn impact on cluster 0:", results)

-------------------------
References
-------------------------
- Financial factor modeling
- Machine learning for finance
- Stress testing in risk management

This module is continuously evolving to incorporate more advanced financial modeling techniques.

