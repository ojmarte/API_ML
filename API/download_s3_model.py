import boto3
import botocore
import re

s3_client = boto3.client('s3')
s3 = boto3.resource('s3')

def download_s3_filemodel(BUCKET_NAME, KEY):
    bucket = s3.Bucket(BUCKET_NAME)
    models = []
    
    for s3_object in bucket.objects.all():
        for key in bucket.objects.all():
            x = re.search("^models/*", key.key)
            if x:
                models.append(key.key)
    
    acc_list = [int(model.split('/')[-1].replace('dt_classifier_acc_', '')) for model in models]
    FOLDER = models[acc_list.index(max(acc_list))]
    FILENAME = re.split(KEY, FOLDER, 1)[1]
    
    try:
        s3_client.download_file(BUCKET_NAME, FOLDER, FILENAME)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise