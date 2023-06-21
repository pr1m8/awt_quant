from rest_framework import serializers
from .models import *

class StocksInsidertradesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksInsidertrades
        fields = ('symbol', 'filing_date', 'transaction_date', 'reporting_cik', 'transaction_type', 'securities_owned', 'company_cik', 'reporting_name', 'type_of_owner', 'acquistion_or_disposition', 'form_type', 'securities_transacted', 'price', 'security_name', 'link',)