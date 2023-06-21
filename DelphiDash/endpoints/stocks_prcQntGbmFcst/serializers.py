from rest_framework import serializers
from .models import *

class StocksPrcQntGbmFcstSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksPrcQntGbmFcst
        fields = ('mean_path', 'median_path', 'path_99', 'path_75', 'path_25', 'path_01', 'min_path', 'max_path', 'stock', 'total_days', 'training_days', 'forecast_days', 'bounded', 'mean', 'increase', 'variance', 'iterations', 'start_date', 'forecast_start_date', 'end_date', 'start_price', 'forecast_median', 'end_price', 'fcst_start_price', 'probability_of_positive_return',)