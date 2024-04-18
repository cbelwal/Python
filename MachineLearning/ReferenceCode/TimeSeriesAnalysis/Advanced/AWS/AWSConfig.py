import boto3

# arn:aws:iam::159491714213:role/ForecastRole
class AWSConfig():
    def __init__(self):
        self.s3_bucket_name = 'dasmicmain' 
        self.region = 'us-east-1'
        self.role_arn = 'arn:aws:iam::159491714213:role/ForecastRole'


    def uploadToS3(self,objectName,filePath):
        # Credentials will be loaded from the %userprofile%\.aws\credentials file
        #s3 = boto3.client('s3', region_name=self.region)
        s3r = boto3.resource('s3', region_name=self.region)
        s3r.Bucket(self.s3_bucket_name).Object(
                objectName).upload_file(filePath)
        
        # Path to your data
        s3_path = f"s3://{self.s3_bucket_name}/{objectName}"
        return s3_path

        