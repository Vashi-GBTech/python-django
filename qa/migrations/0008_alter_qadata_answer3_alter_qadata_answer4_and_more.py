# Generated by Django 4.2.6 on 2023-10-27 10:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('qa', '0007_rename_answer_qadata_answer1_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='qadata',
            name='answer3',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='answer4',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='answer5',
            field=models.TextField(),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='question1',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='question2',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='question3',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='question4',
            field=models.CharField(max_length=255),
        ),
        migrations.AlterField(
            model_name='qadata',
            name='question5',
            field=models.CharField(max_length=255),
        ),
    ]
