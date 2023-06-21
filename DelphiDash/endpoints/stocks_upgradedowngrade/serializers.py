from rest_framework import serializers
from .models import *

class StocksUpgradedowngradeSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksUpgradedowngrade
        fields = ('symbol', 'published_date', 'news_url', 'news_title', 'news_base_url', 'news_publisher', 'new_grade', 'previous_grade', 'grading_company', 'action', 'price_when_posted',)