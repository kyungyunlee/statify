from django import forms 

class ApproveEventForm(forms.Form):
    choices = forms.MultipleChoiceField(widget = forms.CheckboxSelectMultiple())