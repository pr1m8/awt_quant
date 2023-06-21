from rest_framework import serializers
from .models import *

class StocksPrctargetSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksPrctarget
        fields = ('symbol', 'published_date', 'news_url', 'news_title', 'analyst_name', 'price_target', 'adj_price_target', 'price_when_posted', 'news_publisher', 'news_base_url', 'analyst_company',)