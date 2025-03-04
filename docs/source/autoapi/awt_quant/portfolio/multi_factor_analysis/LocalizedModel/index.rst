awt_quant.portfolio.multi_factor_analysis.LocalizedModel
========================================================

.. py:module:: awt_quant.portfolio.multi_factor_analysis.LocalizedModel




Module Contents
---------------

.. py:class:: LocalizedModel(clustered_data, returns_df)

   .. py:attribute:: clustered_data


   .. py:attribute:: returns_df


   .. py:attribute:: results


   .. py:method:: calculate_cluster_returns(cluster_id)


   .. py:method:: plot_residuals(y_test, y_pred, cluster_id)


   .. py:method:: plot_coefficient_importance(factor_significance, cluster_id)


   .. py:method:: train_model_for_cluster(cluster_id)


   .. py:method:: train_all_clusters()


   .. py:method:: perform_time_series_cross_validation(cluster_id, n_splits=5)


