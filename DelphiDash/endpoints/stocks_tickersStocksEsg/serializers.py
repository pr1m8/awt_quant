from rest_framework import serializers
from .models import *

class StocksTickersStocksEsgSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksTickersStocksEsg
        fields = ('group_type', 'group_index', 'date', 'environmentalScore_mean', 'environmentalScore_std', 'environmentalScore_var', 'environmentalScore_median', 'environmentalScore_q25', 'environmentalScore_q75', 'environmentalScore_min', 'environmentalScore_max', 'environmentalScore_count', 'socialScore_mean', 'socialScore_std', 'socialScore_var', 'socialScore_median', 'socialScore_q25', 'socialScore_q75', 'socialScore_min', 'socialScore_max', 'socialScore_count', 'governanceScore_mean', 'governanceScore_std', 'governanceScore_var', 'governanceScore_median', 'governanceScore_q25', 'governanceScore_q75', 'governanceScore_min', 'governanceScore_max', 'governanceScore_count', 'ESGScore_mean', 'ESGScore_std', 'ESGScore_var', 'ESGScore_median', 'ESGScore_q25', 'ESGScore_q75', 'ESGScore_min', 'ESGScore_max', 'ESGScore_count',)