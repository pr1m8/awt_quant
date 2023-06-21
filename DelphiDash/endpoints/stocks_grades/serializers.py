from rest_framework import serializers
from .models import *

class StocksGradesSerializer(serializers.ModelSerializer):
    class Meta:
        model = StocksGrades
        fields = ('symbol', 'date', 'grading_company', 'previous_grade', 'new_grade',)