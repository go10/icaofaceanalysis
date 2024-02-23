import os
import json
import streamlit as st
from PIL import Image
import boto3

class FaceWebApp:
    '''
    Use AWS Rekognition for face image quality checks.
    Streamlit (https://docs.streamlit.io/) is used for the web app.
    '''

    def __init__(self):
        '''
        The AWS credentials are read from local credentials file. 
        https://docs.aws.amazon.com/sdkref/latest/guide/file-location.html 
        Default parameters for the image checks are defined here.
        '''
        self.client = boto3.client('rekognition', region_name='us-east-2')
        st.set_page_config(page_title="Face Image Processing (AWS)", page_icon=":camera:")
        st.title("Face Image Processing (AWS Rekognition)")
        self.conf_is_face = 99.0
        self.min_brightness = 70
        self.min_sharpness = 70
        self.conf_eyeglasses = 80
        self.conf_sunglasses = 80
        self.facepos_left_lo = 0.20
        self.facepos_left_hi = 0.40
        self.facepos_right_lo = 0.60
        self.facepos_right_hi = 0.80
        self.facepos_top_lo = 0.10
        self.facepos_top_hi = 0.25
        self.facepos_bottom_lo = 0.60
        self.facepos_bottom_hi = 0.85
        self.min_image_dim_pixels = 600
        self.min_image_size = 54 * 1024
        self.pose_max_pitchrollyaw = 12.0
        self.conf_mouth_open = 80.0
        self.conf_smile = 70.0
        self.conf_eyes_open = 95.0
        self.conf_eye_dir = 90.0
        self.eye_dir_max_pitchyaw = 10.0
        self.conf_face_occluded = 90.0
    
    def run(self):
        '''
        Display the UI elements on the web page.
        '''
        
        # Upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
        # The max upload size is set in ~/.streamlit/config.toml as [server]\nmaxUploadSize=10
        if uploaded_file is not None:
            save_image_path = './data/' + uploaded_file.name
            with open(save_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(save_image_path, caption='Uploaded Image', width=300)
            
            self.set_parameters()
            
            if st.button("Analyze"):
                # Detect faces
                out_filename = f"data/{uploaded_file.name}.json"
                if os.path.exists(out_filename): # JSON results already exist
                    st.write(f"Using cached JSON: {out_filename}")
                    with open(out_filename, encoding="utf-8") as f:
                        response = json.load(f)
                else: # Call Rekognition
                    st.write("Calling AWS Rekognition")
                    photo = open(save_image_path, 'rb')
                    response = self.client.detect_faces(Image={'Bytes': photo.read()}, 
                                                        Attributes=['ALL'])
                    with open(out_filename, 'w', encoding="utf-8") as f:
                        json.dump(response, f)

                img = Image.open(save_image_path)
                img_size = uploaded_file.size
                self.icao_checks(img_size, img, response)
                self.output_raw_data(img_size, img, response)


    def set_parameters(self):
        '''
        Render the input field to override the parameters for the image checks.
        '''
        self.conf_is_face = st.number_input("is_face_conf", value=self.conf_is_face, 
                                        placeholder=f"{self.conf_is_face}")
        self.min_brightness = st.number_input("min_brightness", value=self.min_brightness, 
                                                placeholder=f"{self.min_brightness}")
        self.min_sharpness = st.number_input("min_sharpness", value=self.min_sharpness,
                                                placeholder=f"{self.min_sharpness}")
        self.conf_eyeglasses = st.number_input("conf_eyeglasses", value=self.conf_eyeglasses,
                                                placeholder=f"{self.conf_eyeglasses}")
        self.conf_sunglasses = st.number_input("conf_sunglasses", value=self.conf_sunglasses,
                                                placeholder=f"{self.conf_sunglasses}")
        self.facepos_left_lo = st.number_input("facepos_left_lo", value=self.facepos_left_lo,
                                                placeholder=f"{self.facepos_left_lo}")
        self.facepos_left_hi = st.number_input("facepos_left_hi", value=self.facepos_left_hi,
                                                placeholder=f"{self.facepos_left_hi}")
        self.facepos_right_lo = st.number_input("facepos_right_lo", value=self.facepos_right_lo,
                                                    placeholder=f"{self.facepos_right_lo}")
        self.facepos_right_hi = st.number_input("facepos_right_hi", value=self.facepos_right_hi,
                                                    placeholder=f"{self.facepos_right_hi}")
        self.facepos_top_lo = st.number_input("facepos_top_lo", value=self.facepos_top_lo,
                                                placeholder=f"{self.facepos_top_lo}")
        self.facepos_top_hi = st.number_input("facepos_top_hi", value=self.facepos_top_hi,
                                                placeholder=f"{self.facepos_top_hi}")
        self.facepos_bottom_lo = st.number_input("facepos_bottom_lo", value=self.facepos_bottom_lo,
                                                    placeholder=f"{self.facepos_bottom_lo}")
        self.facepos_bottom_hi = st.number_input("facepos_bottom_hi", value=self.facepos_bottom_hi,
                                                    placeholder=f"{self.facepos_bottom_hi}")
        self.min_image_dim_pixels = st.number_input("min_image_dim_pixels", 
                                                    value=self.min_image_dim_pixels,
                                                    placeholder=f"{self.min_image_dim_pixels}")
        self.min_image_size = st.number_input("min_image_size", value=self.min_image_size,
                                                placeholder=f"{self.min_image_size}")
        self.pose_max_pitchrollyaw = st.number_input("pose_max_pitchrollyaw", 
                                                     value=self.pose_max_pitchrollyaw,
                                                     placeholder=f"{self.pose_max_pitchrollyaw}")
        self.conf_mouth_open = st.number_input("conf_mouth_open", value=self.conf_mouth_open,
                                                placeholder=f"{self.conf_mouth_open}")
        self.conf_smile = st.number_input("conf_smile", value=self.conf_smile,
                                                placeholder=f"{self.conf_smile}")
        self.conf_eyes_open = st.number_input("conf_eyes_open", value=self.conf_eyes_open,
                                                placeholder=f"{self.conf_eyes_open}")
        self.conf_eye_dir = st.number_input("conf_eye_direction", value=self.conf_eye_dir,
                                            placeholder=f"{self.conf_eye_dir}")
        self.eye_dir_max_pitchyaw = st.number_input("eye_dir_max_pitchyaw", 
                                                    value=self.eye_dir_max_pitchyaw,
                                                    placeholder=f"{self.eye_dir_max_pitchyaw}")
        self.conf_face_occluded = st.number_input("conf_face_occluded", 
                                                  value=self.conf_face_occluded,
                                                  placeholder=f"{self.conf_face_occluded}")
        
        

    def output_raw_data(self, img_size, img, response):
        '''
        Display the raw data from Rekognition.
        '''
        st.header("Raw data:")
        st.write("Image size: ", img_size)
        st.write("Image format: ", img.format)
        st.write("Image mode: ", img.mode)
        st.write("Image size: ", img.size)
        st.write(response)


    def icao_checks(self, img_size, img, resp):
        '''
        Perform the ICAO checks on the image.
        '''
        st.header("Passport Quality / ICAO Checks")

        check_is_face = (resp['FaceDetails'] and 
                         resp['FaceDetails'][0]['Confidence'] > self.conf_is_face)
        st.write(f"0. Is a face(>{self.conf_is_face}): ", ":blue[PASS]" if check_is_face 
                 else ":red[FAIL]")
        if not check_is_face:
            return

        check_shadows_lighting = (resp['FaceDetails'][0]['Quality']['Brightness'] > self.min_brightness 
            and resp['FaceDetails'][0]['Quality']['Sharpness'] > self.min_sharpness)
        st.write(f"1a. Shadows and Lighting(>{self.min_brightness, self.min_sharpness}): ", 
                 ":blue[PASS]" if check_shadows_lighting 
                 else ":red[FAIL]", 
                 " brightness:", resp['FaceDetails'][0]['Quality']['Brightness'],
                 " sharpness:", resp['FaceDetails'][0]['Quality']['Sharpness'])

        check_color = (img.mode == "RGB")
        st.write("1b. Color image(=RGB): ", ":blue[PASS]" if check_color else ":red[FAIL]",
                  img.mode) 

        self.conf_eyeglasses = 80
        check_eyeglasses = (resp['FaceDetails'][0]['Eyeglasses']['Value'] is False 
            and resp['FaceDetails'][0]['Eyeglasses']['Confidence'] > self.conf_eyeglasses)
        st.write(f"2a. No Eyeglasses(>{self.conf_eyeglasses}): ", ":blue[PASS]" if check_eyeglasses 
                 else ":red[FAIL]",
                 " conf:", resp['FaceDetails'][0]['Eyeglasses']['Confidence'])

        limit_2b = 80
        check_sunglasses = (resp['FaceDetails'][0]['Sunglasses']['Value'] is False
                            and resp['FaceDetails'][0]['Sunglasses']['Confidence'] > limit_2b)
        st.write(f"2b. No Sunglasses(>{limit_2b}): ", ":blue[PASS]" if check_sunglasses 
                 else ":red[FAIL]",
                 " conf:", resp['FaceDetails'][0]['Sunglasses']['Confidence'])

        bb_left = resp['FaceDetails'][0]['BoundingBox']['Left']
        bb_top = resp['FaceDetails'][0]['BoundingBox']['Top']
        bb_width = resp['FaceDetails'][0]['BoundingBox']['Width']
        bb_height = resp['FaceDetails'][0]['BoundingBox']['Height']
        bb_right = bb_left + bb_width
        bb_bottom = bb_top + bb_height
        check_face_centered = (self.facepos_left_lo < bb_left and bb_left < self.facepos_left_hi
                 and self.facepos_right_lo < bb_right and bb_right < self.facepos_right_hi
                 and self.facepos_top_lo < bb_top and bb_top < self.facepos_top_hi
                 and self.facepos_bottom_lo < bb_bottom and bb_bottom < self.facepos_bottom_hi)
        st.write("3a. Face Centered: ", ":blue[PASS]" if check_face_centered else ":red[FAIL]")
        st.write(f"    left: {self.facepos_left_lo}<{bb_left}<{self.facepos_left_hi}")
        st.write(f"    right: {self.facepos_right_lo}<{bb_right}<{self.facepos_right_hi}")
        st.write(f"    top: {self.facepos_top_lo}<{bb_top}<{self.facepos_top_hi}")
        st.write(f"    bottom: {self.facepos_bottom_lo}<{bb_bottom}<{self.facepos_bottom_hi}")
        
        check_image_dimensions = (img.size[0] >= self.min_image_dim_pixels and img.size[1] >= self.min_image_dim_pixels)
        st.write(f"4a. Image dimensions (min {self.min_image_dim_pixels}x{self.min_image_dim_pixels}): ",
                  ":blue[PASS]" if check_image_dimensions
                 else ":red[FAIL] ", img.size[0], "x", img.size[1], " pixels")

        check_image_filesize = img_size > self.min_image_size
        st.write(f"4b. Image filesize (min {self.min_image_size}KB): ", ":blue[PASS]" if check_image_filesize 
                 else ":red[FAIL] ", img_size, " bytes")

        check_pose = (abs(resp['FaceDetails'][0]['Pose']['Roll']) < self.pose_max_pitchrollyaw
                      and abs(resp['FaceDetails'][0]['Pose']['Yaw']) < self.pose_max_pitchrollyaw
                      and abs(resp['FaceDetails'][0]['Pose']['Pitch']) < self.pose_max_pitchrollyaw)
        st.write(f"5.1a. Pose(<{self.pose_max_pitchrollyaw}): ", ":blue[PASS]" if check_pose else ":red[FAIL]", 
                 " roll:", resp['FaceDetails'][0]['Pose']['Roll'],
                 " yaw:", resp['FaceDetails'][0]['Pose']['Yaw'],
                 " pitch:", resp['FaceDetails'][0]['Pose']['Pitch'])

        ###limit_52a = 80.0
        ###check_expr_calm = (resp['FaceDetails'][0]['Emotions'][0]['Confidence'] > limit_52a 
        ###                    and resp['FaceDetails'][0]['Emotions'][0]['Type'] == "CALM")
        ###st.write(f"5.2a. Expression: Calm(>{limit_52a})", ":blue[PASS]" if check_expr_calm 
        ###         else ":red[FAIL]",
        ###         " conf:", resp['FaceDetails'][0]['Emotions'][0]['Confidence'])    

        check_expr_mouthopen = (resp['FaceDetails'][0]['MouthOpen']['Value'] is False
                                and resp['FaceDetails'][0]['MouthOpen']['Confidence'] > self.conf_mouth_open)
        st.write(f"5.2b. Expression: Mouth closed(>{self.conf_mouth_open})", ":blue[PASS]" 
                 if check_expr_mouthopen 
                 else ":red[FAIL]", " conf:", resp['FaceDetails'][0]['MouthOpen']['Confidence'])

        check_expr_smile = not (resp['FaceDetails'][0]['Smile']['Value'] is True
                            and resp['FaceDetails'][0]['Smile']['Confidence'] > self.conf_smile)
        st.write(f"5.2c. Expression: No smile(>{self.conf_smile})", ":blue[PASS]" if check_expr_smile 
                 else ":red[FAIL]", " conf:", resp['FaceDetails'][0]['Smile']['Confidence'])

        check_expr_eyesopen = not (resp['FaceDetails'][0]['EyesOpen']['Value'] is False
                            and resp['FaceDetails'][0]['EyesOpen']['Confidence'] > self.conf_eyes_open)
        st.write(f"5.2d. Expression: Eyes open(>{self.conf_eyes_open})", ":blue[PASS]" if check_expr_eyesopen 
                 else ":red[FAIL]", " conf:", resp['FaceDetails'][0]['EyesOpen']['Confidence'])

        check_expr_eyedir = (resp['FaceDetails'][0]['EyeDirection']['Confidence'] > self.conf_eye_dir
                             and abs(resp['FaceDetails'][0]['EyeDirection']['Yaw']) < self.eye_dir_max_pitchyaw
                             and abs(resp['FaceDetails'][0]['EyeDirection']['Pitch']) < self.eye_dir_max_pitchyaw)
        st.write(f"5.2e. Expression: Eye direction(>{self.conf_eye_dir}, <{self.eye_dir_max_pitchyaw})",
                  ":blue[PASS]" 
                 if check_expr_eyedir 
                 else ":red[FAIL]", 
                 " conf:", resp['FaceDetails'][0]['EyeDirection']['Confidence'],
                 " yaw:", resp['FaceDetails'][0]['EyeDirection']['Yaw'],
                 " pitch:", resp['FaceDetails'][0]['EyeDirection']['Pitch'])
        
        check_face_occluded = (resp['FaceDetails'][0]['FaceOccluded']['Value'] is False
                               and resp['FaceDetails'][0]['FaceOccluded']['Confidence'] > self.conf_face_occluded)
        st.write(f"6. Face occluded(>{self.conf_face_occluded}): ", ":blue[PASS]" if check_face_occluded 
                 else ":red[FAIL]", " conf:", resp['FaceDetails'][0]['FaceOccluded']['Confidence'])

        all_checks = (check_is_face
                      and check_shadows_lighting and check_color 
                      and check_eyeglasses and check_sunglasses
                      and check_face_centered 
                      and check_image_dimensions and check_image_filesize
                      and check_pose
                      # and check_expr_calm 
                      and check_expr_mouthopen and check_expr_smile 
                      and check_expr_eyesopen and check_expr_eyedir
                      and check_face_occluded
                    )
        st.write("**All checks :blue[PASS]**" if all_checks 
                 else "**There is a :red[failing] check**")


if __name__ == "__main__":
    app = FaceWebApp()
    app.run()
