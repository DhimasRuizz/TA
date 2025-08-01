# Generated by Django 5.2.1 on 2025-05-31 09:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0003_processeddocument_unique_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='processeddocument',
            name='uploaded_file',
            field=models.FileField(blank=True, null=True, unique=True, upload_to='uploads/'),
        ),
        migrations.AlterField(
            model_name='processeddocument',
            name='unique_id',
            field=models.CharField(default='4d8f91d87fdf4c78b7dfe2d57555a23e', max_length=255, unique=True),
        ),
    ]
