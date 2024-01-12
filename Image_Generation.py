import boto3
from PIL import Image
from io import BytesIO
from Ipython.display import display
import base64
import matplotlib.pyplot as plt
import time

#helper decoder
def decode_base64_image(image_string):
    base64_image = base64.b64encode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)

#display PIL image as grid
def display_image(images=None, columns=3, width=100, height=100):
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns +1), columns, i + 1)
        plt.axis('off')
        plt.imshow(image)

start_time = time.time()
#run prediction

client = boto3.client('sagemaker-runtime')
prompt = "A dog trying to catch a flying pizza art"
num_images_per_prompt = 1
payload = {
    "inputs": prompt,
    "num_images_per_prompt": num_images_per_prompt 
}
serialized_payload = json.dumps(payload)

endpoint_name = "jumpstart-dft-stabilityai-stable-di-20240112-201003"
response = client.invoke_endpoint(
	EndpointName=endpoint_name,
    Body=serialized_payload 
	ContentType='application/json', 
	Accept='application/json;jpeg', 
	Body=json.dumps(payload))

from PIL import Image
import numpy as np

def parse_response(query_response):
    response_dict = json.loads(query_response)
    return response_dict["generated_images"], response_dict["prompt"]
    
response_payload = response['Body'].read().decode('utf-8')
generated_images, prompt = parse_response(response_payload)
        
image = Image.fromarray(np.uint8(generated_images[0]))
buffer = io.BytesIO()
image.save(buffer, "jpeg")
buffer.seek(0)
            
s3 = boto3.client('s3')
s3.upload_fileobj(buffer, mybucket, mykey, ExtraArgs={"ContentType": "image/jpeg"})