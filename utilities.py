# Required Imports
from stegano import lsb
import cv2
import numpy as np
from PIL import ImageEnhance, Image
from cv2 import dnn_superres
import dlib
import pytesseract as pyt
import pyautogui
from datetime import datetime

THRESHOLD_EAR_VALUE = 0.27
captured_ss_path = ""
convexhull = None
frame_to_capture = None

# For steganography
class Steganographer:
    def __init__(self, file):
        self.__file = file
        self.__uploaded_file_path = f"./static/uploads/{self.__file}.png"
        self.__download_file_path = f"./static/downloads/{self.__file}_encryptedByStegano.png"

    def hide(self, msg):
        secret = lsb.hide(self.__uploaded_file_path, msg)
        secret.save(self.__download_file_path)
        return self.__download_file_path

    def reveal(self):
        reveal_txt = lsb.reveal(self.__uploaded_file_path)
        return reveal_txt

# For text extraction
class TextExtractor:
    def __init__(self, file):
        self.__file = file
        
    def extract_txt(self):
        """ Extract text from images """
        img = cv2.imread(self.__file)

        # Convert image into Grayscale and then apply threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # Kernel to detect sentences (around rectangle)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Dilation to create enlarged image of same shape
        dilated = cv2.dilate(threshold, kernel, 1)
            
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,  
                                    cv2.CHAIN_APPROX_NONE) 
        text = ""
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt) # Get the coordinated where txt is located 
            cropped_txt_img = img[y:y+h, x:x+w] 
            text += pyt.image_to_string(cropped_txt_img)
            
        # " \n\x0c" is a string that is returned when the img does not contain any text
        print(text) 
        print()
        if text != " \n\x0c":
            return text

# For enhancing resolution
class ImageProcessor:
    def __init__(self, file):
        self.__file = file
        self.__uploaded_file_path = f"./static/uploads/{self.__file}"
        self.__model_path = "./model/FSRCNN_x4.pb"

        filename, ext = self.__file.split(".")
        self.__download_file_path = f"./downloads/{filename}_super_resolution.{ext}"

    def enhance_res(self):
        """ Enhance resolution of image using pre-trained model""" 
        img = cv2.imread(self.__uploaded_file_path)

        # Apply DNN Super Resolution technique using pre trained model (in our case FSRCNN model)
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(self.__model_path)
        sr.setModel("fsrcnn", 4)

        final_img = sr.upsample(img)

        final_img = cv2.fastNlMeansDenoisingColored(final_img, None, 10, 10, 7, 15) # Remove Noise
        cv2.imwrite(self.__download_file_path, final_img)
        return self.__download_file_path

class Filters:
    def __init__(self, switch):
        self.__switch = switch
        self.__download_folder = "./static/downloads"
    
    def generate_frames(self, grey, negative, cartoon, water_color,
                      pencil, phantom, blur, capture_ss):
        """ Generates the captured frames and 
        yields them as bytes """
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            if self.__switch[0]:
                res, frame = cap.read()
                if res:
                    if grey[0]:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if negative[0]:
                        frame = cv2.bitwise_not(frame)

                    if cartoon[0]:
                        smooth_img = cv2.bilateralFilter(frame, 10, 250, 250)
                        # Work on edge lines
                        gray = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2GRAY)
                        img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
                        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY, 9, 10)

                        # Multiply the original and edge lined img
                        frame = cv2.bitwise_and(smooth_img, smooth_img, mask=img_edge)

                    if water_color[0]:
                        frame = cv2.stylization(frame, sigma_s=110, sigma_r=0.15)

                    if pencil[0]:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        inverted_img = cv2.bitwise_not(gray)
                        smooth_img = cv2.GaussianBlur(inverted_img, (21,21), sigmaX=0, sigmaY=0)

                        frame = cv2.divide(gray, 255 - smooth_img, scale=256)

                    if phantom[0]:
                        _, frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY_INV)
                    
                    if blur[0]:
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
                        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                        blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
                        frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)

                    if capture_ss[0]:
                        global captured_ss_path
                        curr_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                        captured_ss_path = f"{self.__download_folder}/image_vision_shot{curr_time}.jpg"
                        cv2.imwrite(captured_ss_path, frame)
                        capture_ss[0] = False

                    try:
                        _, buffer = cv2.imencode(".jpg", cv2.flip(frame,1))
                        frame = buffer.tobytes()
                        yield (b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

                    except Exception:
                        pass
            else:
                cap.release()
                cv2.destroyAllWindows()

class FaceSwapper:
    def __init__(self, swap_face_file, stream_on, stream_off):
        self.__stream_on = stream_on
        self.__stream_off = stream_off
        self.__swap_face_file = swap_face_file
        self.__download_folder = "./static/downloads"
        self.__predictor_model = "./model/shape_predictor_68_face_landmarks.dat"
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(self.__predictor_model)

    def initiate_swap(self):
        self.img = cv2.imread(self.__swap_face_file)
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(img_gray)
        
        indexes_triangles = []
        faces = self.__detector(img_gray)
        for face in faces:
            landmarks = self.__predictor(img_gray, face)
            self.landmark_points = []
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                self.landmark_points.append((x, y))

            points = np.array(self.landmark_points, np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # Delaunay Triangulation
            rect = cv2.boundingRect(hull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(self.landmark_points)

            triangle_list = np.array(subdiv.getTriangleList(), np.int32)

            for t in triangle_list:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                index_pt1 = np.where((points == pt1).all(axis=1))
                index_pt1 = self.extract_index(index_pt1)

                index_pt2 = np.where((points == pt2).all(axis=1))
                index_pt2 = self.extract_index(index_pt2)

                index_pt3 = np.where((points == pt3).all(axis=1))
                index_pt3 = self.extract_index(index_pt3)

                if index_pt1 and index_pt2 and index_pt3:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    indexes_triangles.append(triangle)

        return indexes_triangles
    
    def extract_index(self, nparray):
        for num in nparray[0]:
            index = num
            return index

    def generate_frames(self, capture_ss):
        """ Swaps the faces and generates 
        the captured frames and yields them as bytes """
        global frame_to_capture
        if self.__swap_face_file and self.__stream_on[0]:
            cap = cv2.VideoCapture(0)
            print("Recording...")
            while cap.isOpened():
                _, frame = cap.read()
                # img_frame = frame
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                new_img = np.zeros_like(img_gray)

                faces = self.__detector(img_gray)
                if faces:
                    global convexhull
                    for face in faces:
                        landmarks = self.__predictor(img_gray, face)
                        landmark_points = []
                        
                        for n in range(68):
                            x = landmarks.part(n).x
                            y = landmarks.part(n).y
                            landmark_points.append((x, y))

                        points = np.array(landmark_points, np.int32)
                        convexhull = cv2.convexHull(points)
                    
                    indexes_triangles = self.initiate_swap()
                    for idx in indexes_triangles:
                        tr1_pt1 = self.landmark_points[idx[0]]
                        tr1_pt2 = self.landmark_points[idx[1]]
                        tr1_pt3 = self.landmark_points[idx[2]]
                        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                        rect1 = cv2.boundingRect(triangle1)
                        (x, y, w, h) = rect1
                        cropped_triangle = self.img[y: y + h, x: x + w]
                        cropped_tr1_mask = np.zeros((h, w), np.uint8)

                        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

                        # Triangulation of second face
                        tr2_pt1 = landmark_points[idx[0]]
                        tr2_pt2 = landmark_points[idx[1]]
                        tr2_pt3 = landmark_points[idx[2]]
                        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                        rect2 = cv2.boundingRect(triangle2)
                        (x, y, w, h) = rect2

                        cropped_tr2_mask = np.zeros((h, w), np.uint8)

                        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                        points = np.float32(points)
                        points2 = np.float32(points2)
                        affine_transform = cv2.getAffineTransform(points, points2)
                        warped_triangle = cv2.warpAffine(cropped_triangle, affine_transform, (w, h))
                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                        # Reconstructing destination face
                        img2_new_face_rect_area = new_img[y: y + h, x: x + w]
                        # img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area, 1, 255, cv2.THRESH_BINARY_INV)
                        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
                        warped_triangle = cv2.cvtColor(warped_triangle, cv2.COLOR_BGR2GRAY)
                        # print(img2_new_face_rect_area.shape, warped_triangle.shape)
                        # print()
                        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                        new_img[y: y + h, x: x + w] = img2_new_face_rect_area

                        img2_face_mask = np.zeros_like(img_gray)
                        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull, 255)
                        img2_face_mask = cv2.bitwise_not(img2_head_mask)

                        img2_head_noface = cv2.bitwise_and(frame, frame, mask=img2_face_mask)
                        img2_head_noface = cv2.cvtColor(img2_head_noface, cv2.COLOR_BGR2GRAY)
                        result = cv2.add(img2_head_noface, new_img)

                        (x, y, w, h) = cv2.boundingRect(convexhull)
                        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

                        seamlessclone = cv2.seamlessClone(result, frame, img2_head_mask, center_face2, cv2.MIXED_CLONE)
                        frame_to_capture = seamlessclone

                        try:
                            _, buffer = cv2.imencode(".jpg", cv2.flip(seamlessclone, 1))
                            to_stream = buffer.tobytes()
                            yield (b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n" + to_stream + b"\r\n")
                        
                        except Exception as e:
                            print(e)

                if self.__stream_off[0]:
                    cap.release()
                    cv2.destroyAllWindows()

                if capture_ss[0]:
                    global captured_ss_path
                    curr_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                    captured_ss_path = f"{self.__download_folder}/image_vision_face_swap{curr_time}.jpg"
                    cv2.imwrite(captured_ss_path, frame_to_capture)
                    capture_ss[0] = False

class FaceControl:
    def __init__(self, stream_on, stream_off):
        self.__stream_on = stream_on
        self.__stream_off = stream_off 
        self.__classifier_path = "./model/haarcascade_eye_tree_eyeglasses.xml"
        self.__hog_face_detector = dlib.get_frontal_face_detector()
        self.__dlib_facelandmark = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
        self.__classifier = cv2.CascadeClassifier(self.__classifier_path)

    def get_single_eye_cord(self, eyes):
        """ Fetches the coordinates of left eye """
        if eyes[0].any():
            return eyes[0][0], eyes[0][1], eyes[0][2], eyes[0][3]

    def calculate_EAR(self, eye):
        """ Function responsible for calculating
        Eye Aspect Ratio (EAR) """
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        eye_aspect_ratio = (A+B)/(2.0*C)
        return eye_aspect_ratio

    def generate_frames(self):
        cap = cv2.VideoCapture(0)
        w = cap.get(3)
        h = cap.get(4)

        pyautogui.FAILSAFE = False
        display_dim = pyautogui.size()
        x_cord, y_cord = display_dim[0] // 2, display_dim[1] // 2

        if self.__stream_on[0]:
            while cap.isOpened():
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                eyes = self.__classifier.detectMultiScale(gray, 1.2, 5)
                faces = self.__hog_face_detector(gray)
                for _ in eyes:
                    lx, ly, lw, lh = self.get_single_eye_cord(eyes)
                    cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), (0, 255, 0), 2)
                    print(lx, ly)
                    
                    x_cord = np.interp(lx, (0, w), (0, display_dim[0]))
                    y_cord = np.interp(ly, (0, h), (0, display_dim[1]))

                    pyautogui.moveTo(x_cord, y_cord, 0)

                for face in faces:
                    face_landmarks = self.__dlib_facelandmark(gray, face)
                    left_eye = []
                    right_eye = []

                    for n in range(36,42): # For right eye detection
                        x1 = face_landmarks.part(n).x
                        y1 = face_landmarks.part(n).y
                        left_eye.append((x1,y1))

                        # Draw outline of eye
                        # next_point = n+1
                        # if n == 41:
                        #     next_point = 36
                        # x2 = face_landmarks.part(next_point).x
                        # y2 = face_landmarks.part(next_point).y
                        # # cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 1)

                    for n in range(42,48): # For left eye detection
                        x1 = face_landmarks.part(n).x
                        y1 = face_landmarks.part(n).y
                        right_eye.append((x1,y1))

                        # Draw outline of eye 
                        # next_point = n+1
                        # if n == 47:
                        #     next_point = 42
                        
                        # x2 = face_landmarks.part(next_point).x
                        # y2 = face_landmarks.part(next_point).y
                        # cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 1)

                    left_EAR = self.calculate_EAR(left_eye)
                    right_EAR = self.calculate_EAR(right_eye)

                    final_EAR = (left_EAR + right_EAR) / 2
                    # print(final_EAR, left_EAR)
                    if round(right_EAR, 2) < THRESHOLD_EAR_VALUE:
                        pyautogui.click(button="right")

                    if round(final_EAR, 2) < THRESHOLD_EAR_VALUE:
                        pyautogui.click(button="left")
        
                try:
                    _, buffer = cv2.imencode(".jpg", frame)
                    to_stream = buffer.tobytes()
                    yield (b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + to_stream + b"\r\n")
                
                except Exception as e:
                    print(e)
                
                if self.__stream_off[0]:
                    cap.release()
            cv2.destroyAllWindows()

class Editor:
    def __init__(self, file):
        self.__file = file
        self.__uploaded_file_path = f"./static/uploads/{self.__file}"

        filename, ext = self.__file.split(".")
        self.__download_file_path = f"./static/downloads/{filename}_edited.{ext}"

    def process_img(self, img_attribs):
        '''Alter the image based on image attributes array '''
        brightness, contrast, sharpness, width, \
            height, r_value, g_value, b_value = img_attribs
        is_edited = False

        img = Image.open(self.__uploaded_file_path)

        if brightness != 1:
            img = ImageEnhance.Brightness(img).enhance(brightness)
            is_edited = True
        
        if contrast != 1:
            img = ImageEnhance.Contrast(img).enhance(contrast)
            is_edited = True
        
        if sharpness != 1:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
            is_edited = True

        img = np.array(img).astype(np.float)

        if height and width:
            print(height, width)
            img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            is_edited = True
        
        if b_value != 1:
            img[..., 0] *= np.clip(img[:, :, 0] * b_value, a_min=0, a_max=255)
            is_edited = True
        
        if g_value != 1:
            img[..., 1] *= np.clip(img[:, :, 1] * g_value, a_min=0, a_max=255)
            is_edited = True
        
        if r_value != 1:
            img[..., 2] *= np.clip(img[:, :, 2] * r_value, a_min=0, a_max=255)
            is_edited = True

        print("Editied", is_edited, self.__download_file_path)
        if is_edited:
            (b, g, r) = cv2.split(img)
            img = cv2.merge([r, g, b])
            cv2.imwrite(self.__download_file_path, img)

        return self.__download_file_path

        



                
                
