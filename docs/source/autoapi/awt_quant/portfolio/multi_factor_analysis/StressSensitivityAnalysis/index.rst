awt_quant.portfolio.multi_factor_analysis.StressSensitivityAnalysis
===================================================================

.. py:module:: awt_quant.portfolio.multi_factor_analysis.StressSensitivityAnalysis




Module Contents
---------------

.. py:class:: StressAnalysis(localized_model, clustered_data)

   .. py:attribute:: localized_model


   .. py:attribute:: clustered_data


   .. py:method:: noise_injection(cluster_id, noise_level=0.05)


   .. py:method:: extreme_value_analysis(cluster_id, feature, extreme='max')


   .. py:method:: global_downturn(cluster_id, downturn_pct=0.1)


   .. py:method:: rapid_inflation(cluster_id, inflation_factors, inflation_pct=0.2)


.. py:class:: SensitivityAnalysis(localized_model, clustered_data)

   .. py:attribute:: localized_model


   .. py:attribute:: clustered_data


   .. py:method:: feature_perturbation(cluster_id, feature, perturb_pct=0.05)

      Perturb one feature and evaluate the change in model predictions.



   .. py:method:: feature_importance_analysis(cluster_id)


