from face_alignment import align
from inference import *
from PIL import Image as PILImage
import io
from IPython.display import Image
import IPython.display as display
import time
import cv2
from multiprocessing.pool import ThreadPool


VIDEO_PATH = 'test_videos/IRON MAN 2 (2010) .mp4'
THRESHOLD = 0.5
NUM_CPUS = cv2.getNumberOfCPUs()

# Load model
model = load_pretrained_model_w_cuda('ir_101')
# feature, norm = model(torch.randn(2,3,112,112))
tensor = torch.randn(2,3,112,112)
feature, norm = model(tensor.to(torch.device('cuda:0')))

def process_frame(frame, frame_info_list):
    for each_face_info in frame_info_list:
        name = each_face_info['name'].split('.', 1)[0]
        bounding_box = each_face_info['bounding_box']
        x1, y1, x2, y2, _ = bounding_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name,(x1+6,y2-6), font, 1.0 , (255,255,255), 1)
    return frame
        
    

# Load database features
test_image_path = 'face_alignment/test_images'
features = []
names = []
database_info_list = []
database_features = []
for fname in sorted(os.listdir(test_image_path)):
    # print(fname)
    path = os.path.join(test_image_path, fname)
    aligned_rgb_img, bounding_box = align.get_aligned_face(path, source="database")
    bgr_tensor_input = to_input_w_cuda(aligned_rgb_img)
    bgr_tensor_input_w_cuda = bgr_tensor_input.cuda()
    feature, _ = model(bgr_tensor_input_w_cuda)
    # print(bgr_tensor_input)
     
    each_face_info = {
        'name': fname,
        'source': 'database',
        'face': aligned_rgb_img,
        'feature': feature,
        'bounding_box': bounding_box[0],
        'type': 'NIL'
    }
    database_info_list.append(each_face_info)
    database_features.append(feature)
    # print(len(feature[0]))
    names.append(fname)

num_database_faces = len(database_info_list)
# print(database_info_list)

# print(len(database_features))
# print(len(database_features[0]))
# print(len(database_features[0][0]))

# Open a connection to the default camera
video_capture = cv2.VideoCapture(VIDEO_PATH)

if not video_capture.isOpened():
    print("Error: Could not open video stream.")
    exit()

frame_count = 0

try:
    while True:

        # pool = ThreadPool(processes = NUM_CPUS)
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to an image that can be displayed in Jupyter
        pil_img = PILImage.fromarray(frame.copy())
        with io.BytesIO() as f:
            pil_img.save(f, format='jpeg')
            display_img = Image(data=f.getvalue())
        
        # Obtain frame features
        frame_info_list = []
        features = database_features.copy()
        
        aligned_rgb_imgs, bounding_boxes = align.get_aligned_face("None", source="frame", rgb_pil_image=pil_img)
        if aligned_rgb_imgs:
            time1 = time.time()
            for i, aligned_rgb_img in enumerate(aligned_rgb_imgs):
                print("face found")
                time2 = time.time()
                bounding_box = bounding_boxes[i]
                bgr_tensor_input = to_input_w_cuda(aligned_rgb_img)
                time3 = time.time()
                # feature, _ = model(bgr_tensor_input)
                feature, _ = model(bgr_tensor_input.cuda())
                time4 = time.time()
                each_face_info = {
                    'name': '?',
                    'source': 'frame',
                    'face': aligned_rgb_img,
                    'feature': feature,
                    'bounding_box': bounding_box,
                    'type': '?'
                }
                
                frame_info_list.append(each_face_info)
                features.append(feature)
                names.append(i)
                time5 = time.time()
            # print(frame_info_list)
            # print(len(feature[0]))
            # features.append(feature)

            # print(features)
            # print(len(features))
            # print(features[0])
            # print(len(features[0]))

            # Compare database and frame faces
            similarity_scores = torch.cat(features) @ torch.cat(features).T
            similarty_scores_list = similarity_scores.tolist()
            print()
            print(f"score: {similarity_scores}")
            print()
            # Update frame file infos
            for i, file_info in enumerate(frame_info_list):
                scorelist = []
                k = i + num_database_faces
                for j in range(num_database_faces):
                    scorelist.append(similarty_scores_list[j][k])
                    max_score = max(scorelist)
                    if max_score >= THRESHOLD:
                        database_index = scorelist.index(max_score)
                        file_info['name'] = database_info_list[database_index]['name']
                print(scorelist)
            # print(frame_info_list)
            print(time2-time1, time3-time2, time4-time3, time5-time4)
            processed_frame = process_frame(frame, frame_info_list)
            save_path = rf'processed_frames/frame{frame_count}.jpg'
            # cv2.imwrite(save_path, processed_frame)
            frame_count += 1
        else:
            processed_frame = frame.copy()

        # Convert the frame to an image that can be displayed in Jupyter
        pil_img = PILImage.fromarray(processed_frame.copy())
        with io.BytesIO() as f:
            pil_img.save(f, format='jpeg')
            display_img = Image(data=f.getvalue())
            
        # Display the frame
        display.clear_output(wait=True)
        display.display(display_img)



finally:
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
