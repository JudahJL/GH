from django import forms


class GarbageForm(forms.Form):
    img = forms.ImageField()