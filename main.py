import cv2
import numpy

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                cameraMatrix = matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
    if len(corners) > 0:
        for i in range(0, len(ids)):
            rec, tec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                          distortion_coefficients)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rec, tec, 0.01)
    return frame


def aruco_display(corners, ids, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            cv2.line(image, top_left, top_right, (0, 255, 0), 2)
            cv2.line(image, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(image, bottom_left, top_left, (0, 255, 0), 2)
            c_x = int((top_left[0] + bottom_right[0]) / 2.0)
            c_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(image, (c_x, c_y), 4, (0, 0, 255), -1)
            cv2.putText(image, str(markerID), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
    return image


def generate_marker():
    x, y = input('Type: ').split(' ')
    aruco_type = 'DICT_'+x+'X'+x+'_'+y
    m_id = int(input('Marker ID: '))
    tag_size = int(input('Tag Size: '))
    tag = numpy.zeros((tag_size, tag_size, 1), dtype='uint8')
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    cv2.aruco.drawMarker(aruco_dict, m_id, tag_size, tag, 1)
    tag_name = input('Tag Name: ')
    dr = 'ArUcoMarker/' + tag_name + '.png'
    cv2.imwrite(dr, tag)
    print('Generated')
    image_read(dr)


def image_read(file_name):
    img = cv2.imread(file_name)
    aruco_type = "DICT_5X5_1000"
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    detected_markers = aruco_display(corners, ids, img)
    cv2.imshow('Image', detected_markers)
    cv2.waitKey(0)
    cv2.distroyAllWindows()


def webcam_read():
    aruco_type = "DICT_5X5_100"
    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    aruco_params = cv2.aruco.DetectorParameters_create()
    vid = cv2.VideoCapture(0)
    while vid.isOpened():
        ret, img = vid.read()
        h, w, d = img.shape
        width = 1000
        height = int(width*(h/w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        detected_markers = aruco_display(corners, ids, img)
        cv2.imshow('WebCam', detected_markers)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.distroyAllWindows()


def video_read():
    vid = cv2.VideoCapture('testVideo.mp4')
    while vid.isOpened():
        ret, frame = vid.read()
        if ret > 0:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.releaseAllWindows()


def position_estimation():
    aruco_type = "DICT_5X5_100"
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    intrinsic_camera = numpy.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
    distortion = numpy.array((-0.43948, 0.18514, 0, 0))

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
        cv2.imshow('Estimated Pose', output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


n = int(input('Enter: '))
if n == 1:
    print('Image')
    image_read('test.jpg')
elif n == 2:
    print('WebCam')
    webcam_read()
elif n == 3:
    print('Videos')
    video_read()
elif n == 4:
    print('Position Estimation')
    position_estimation()
else:
    print('Generate Aruco')
    generate_marker()

