from django.db import models, transaction


class TimestampModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class SortableModel(models.Model):
    order = models.PositiveSmallIntegerField(default=0, blank=False, null=False)

    class Meta:
        abstract = True

    def get_ordering_queryset(self):
        raise NotImplementedError("Unknown ordering queryset")

    def get_max_order(self, qs):
        existing_max = qs.aggregate(models.Max("order"))
        existing_max = existing_max.get("order__max")
        return existing_max
        
    def save(self, *args, **kwargs):
        if self.pk is None and not self.order:
            qs = self.get_ordering_queryset()
            existing_max = self.get_max_order(qs)
            self.order = 0 if existing_max is None else existing_max + 1
        super().save(*args, **kwargs)

    @transaction.atomic
    def delete(self, *args, **kwargs):
        if self.order is not None:
            qs = self.get_ordering_queryset()
            qs.filter(order__gt=self.order).update(
                order=models.F("order") - 1
            )
        super().delete(*args, **kwargs)
