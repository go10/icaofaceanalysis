import json
import boto3

class RekognitionAccess:
    '''
    https://docs.aws.amazon.com/code-library/latest/ug/python_3_rekognition_code_examples.html 
    https://docs.aws.amazon.com/rekognition/latest/dg/identity-verification-tutorial.html
    AWS credentials are stored in local credentials file.
    '''

    def __init__(self):
        self.client = boto3.client('rekognition', region_name='us-east-1')

    def check_face(self, target_file):
        image_target = open(target_file, 'rb')
        response = self.client.detect_faces(Image={'Bytes': image_target.read()},
                                            Attributes=['ALL'])
        print('Detected faces for ' + target_file)

        for face_detail in response['FaceDetails']:
            print('The detected face is between ' + str(face_detail['AgeRange']['Low'])
                + ' and ' + str(face_detail['AgeRange']['High']) + ' years old')

            print('Here are the other attributes:')
            print(json.dumps(face_detail, indent=4, sort_keys=True))

            # Access predictions for individual face details and print them
            print("Gender: " + str(face_detail['Gender']))
            print("Smile: " + str(face_detail['Smile']))
            print("Eyeglasses: " + str(face_detail['Eyeglasses']))
            print("Emotions: " + str(face_detail['Emotions'][0]))

        return len(response['FaceDetails'])


# For testing...
if __name__ == "__main__":
    rek = RekognitionAccess()

    PHOTO = "data/face185.jpeg"
    face_count = rek.check_face(PHOTO)
    print("Faces detected: " + str(face_count))

    if face_count == 1:
        print("Image suitable for use in collection.")
    else:
        print("Please submit an image with only one face.")
