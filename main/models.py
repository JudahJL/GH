from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from django.db.models import Q


# Create your models here.
# kinda undecided
class TrashReport(models.Model):
    ...

    class Meta:
        abstract = True


# Never remove Status, only add
class TrashTicket(models.Model):
    class Status(models.IntegerChoices):
        OPEN = 1
        RESOLVED = 2
        CLEAN = 3
        FALSE_ALARM = 4
        CLEARED = 5
        URGENT = 6

    pincode = models.IntegerField()
    status = models.IntegerField(null=False, blank=False, choices=Status)
    first_reported = models.DateField(null=False, blank=False)
    last_seen = models.DateField(null=False, blank=False)
    severity = models.PositiveSmallIntegerField(null=False, blank=False, validators=[MinValueValidator(0), MaxValueValidator(10)])
    action = models.TextField(null=True, blank=True)
    image = models.ImageField(null=False, blank=False)
    latitude = models.FloatField(null=False, blank=False)
    longitude = models.FloatField(null=False, blank=False)
    camera_model = models.TextField(null=True, blank=True)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=Q(severity__gte=0) & Q(severity__lte=10),
                name="severity_between_0_and_10",
            ),
        ]
