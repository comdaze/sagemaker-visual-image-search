# locally
# docker run -v -d -p 8080:8080 image-embedding
import requests
import json
url='http://localhost:8080/invocations'
bucket = 'sagemaker-cn-northwest-1-343958593302'
image_uri = 'image-embedding/1.jpg'
test_data = {
    'bucket' : bucket,
    'image_uri' : image_uri,
    'content_type': "application/json",
}
payload = json.dumps(test_data)
r = requests.post(url,data=payload)
print(r.json())

# on sagemaker
# python create_endpoint
import boto3
from botocore.config import Config
from boto3.session import Session
import json

config = Config(
    read_timeout=120,
    retries={
        'max_attempts': 0
    }
)

def infer(input_image):
    bucket = 'sagemaker-cn-northwest-1-343958593302'
    image_uri = input_image
    test_data = {
        'bucket' : bucket,
        'image_uri' : image_uri,
        'content_type': "application/json",
    }
    payload = json.dumps(test_data)
    print(payload)

    sagemaker_runtime_client = boto3.client('sagemaker-runtime', config=config)
    session = Session(sagemaker_runtime_client)

#     runtime = session.client("runtime.sagemaker",config=config)
    response = sagemaker_runtime_client.invoke_endpoint(
        EndpointName='image-embedding',
        ContentType="application/json",
        Body=payload)

    result = json.loads(response["Body"].read())
    print (result)

infer('image-embedding/1.jpg')