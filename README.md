Sample code to do ICAO checks on a facial image using cloud services.
It uses Streamlit (https://streamlit.io) for the web UI.

face_webapp_aws.py uses AWS Rekognition. To run, you will need an AWS account
with the permission to make calls to Rekognition. The AWS credentials are 
by default read from your local filesystem in the default location (~/.aws/credentials).

face_webapp_azure.py uses Azure AI services. You will need an Azure account with
the permissions to make calls to the Cognitive Services API. The Azure credentials
are read from the two environment variables, VISION_KEY and VISION_ENDPOINT.

Software license is in the LICENSE file.
