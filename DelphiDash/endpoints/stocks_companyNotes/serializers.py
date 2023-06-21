from rest_framework import serializers
from .models import *

class StocksCompanyNotesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksCompanyNotes
        fields = ('symbol', 'cik', 'title', 'exchange',)