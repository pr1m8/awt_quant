from rest_framework import serializers
from .models import *

class StocksRatingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksRatings
        fields = ('symbol', 'date', 'rating', 'rating_score', 'rating_recommendation', 'rating_details_dcf_score', 'rating_details_dcf_recommendation', 'rating_details_roe_score', 'rating_details_roe_recommendation', 'rating_details_roa_score', 'rating_details_roa_recommendation', 'rating_details_de_score', 'rating_details_de_recommendation', 'rating_details_pe_score', 'rating_details_pe_recommendation', 'rating_details_pb_score', 'rating_details_pb_recommendation',)