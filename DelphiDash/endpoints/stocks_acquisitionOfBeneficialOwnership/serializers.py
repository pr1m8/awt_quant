from rest_framework import serializers
from .models import *

class StocksAcquisitionOfBeneficialOwnershipSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksAcquisitionOfBeneficialOwnership
        fields = ('symbol', 'cik', 'filing_date', 'accepted_date', 'cusip', 'name_of_reporting_person', 'citizenship_or_place_of_organization', 'sole_voting_power', 'shared_voting_power', 'sole_dispositive_power', 'shared_dispositive_power', 'amount_beneficially_owned', 'percent_of_class', 'type_of_reporting_person', 'url',)