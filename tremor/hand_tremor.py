from enum import Enum
import yaml
import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import mediapipe as mp
data = {"UPDRS":None}
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
class Tremor(Enum):
    Resting = 0
    Postural = 1
# Camera parameters:
CAMERA_FOCAL_LENGTH = None  
CAMERA_FOCAL_LENGTH_STD = None 
CAMERA_NATIVE_ASPECT = None  
CAMERA_VIDEO_ASPECT = None  
# Video file:
VIDEO_FILEPATH = None
videoFilename = None
videoWidth = None  
videoFramerate = None  


START_FRAME = None
END_FRAME = None
HAND_DEPTH = None

#GUI
GUI_HAND_TRACKING = True


AUTO_MODE = None


tremorType = None


chosenLandmarks = None
chosenLandmarksText = None
chosenLandmarksID = None

USE_CUSTOM_LANDMARKS = None


capture = None

SHOW_PLOT_LEGEND = None
def calcUpdrsRating(amplitude, totalAmplitudeError):
    if (amplitude - totalAmplitudeError) <= 0.01:
        return 0
    elif amplitude < 1:
        return 1
    elif amplitude < 3:
        return 2
    elif amplitude < 10:
        return 3
    else:
        return 4


def calcErrorFromHandTracking(amplitude):
    avg = sum(amplitude) / len(amplitude)

    rmse = 0
    for a in amplitude:
        rmse += (a - avg) ** 2
    rmse = math.sqrt(rmse / len(amplitude))
    return rmse / 2
def calcPixelDistAmplitudeFromPath(path):
    pathDerivative = []
    for i in range(0, len(path) - 1):
        # https://math.stackexchange.com/questions/304069/use-a-set-of-data-points-from-a-graph-to-find-a-derivative
        gradientAtPoint = path[i+1] - path[i]
        pathDerivative.append(gradientAtPoint)
    peaksAndTroughs = []
    lastStepGradIsPos = pathDerivative[0] >= 0
    for i in range(1, len(pathDerivative) - 3):
        if pathDerivative[i] == 0:
            continue
        if lastStepGradIsPos != (pathDerivative[i] >= 0):

            if lastStepGradIsPos != (pathDerivative[i+3] >= 0):
                peaksAndTroughs.append(path[i])
                lastStepGradIsPos = not lastStepGradIsPos
    amplitudeSamples = []
    for i in range(0, len(peaksAndTroughs) - 1):
        diff = abs(peaksAndTroughs[i] - peaksAndTroughs[i+1])
        amplitudeSamples.append(diff)
    stdDv = np.std(amplitudeSamples)
    amplitudeMedian = np.median(amplitudeSamples)
    filteredAmplitudeSamples = filter(lambda ampl:
                                      ampl <= amplitudeMedian + (2 * stdDv),
                                      amplitudeSamples)
    amplitude2SdMax = max(filteredAmplitudeSamples)
    return amplitudeMedian, amplitude2SdMax
def selectLandmarks():
    global chosenLandmarks, chosenLandmarksText, chosenLandmarksID

    if USE_CUSTOM_LANDMARKS:
        chosenLandmarksText = 'User specified landmarks; see config file'
        chosenLandmarksID = 'custom'
        return

    if tremorType == Tremor.Resting:
        if AUTO_MODE:
            inp = str(sys.argv[4]).upper()[0]
        else:
            print('Track MCP joints (first knuckles), PIP joints, DIP joints'
                  + ' or tips of fingers?')
            # inp = input('> Type in MCP, PIP, DIP, or TIP: ').upper()[0]
            inp = 'T'
        if inp == 'M':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_MCP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                               mp_hands.HandLandmark.INDEX_FINGER_MCP]
            chosenLandmarksText = 'Index, middle and ring finger MCP joints'
            chosenLandmarksID = 'MCP'
        elif inp == 'P':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_PIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP]
            chosenLandmarksText = 'Index, middle and ring finger PIP joints'
            chosenLandmarksID = 'PIP'
        elif inp == 'D':
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_DIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                               mp_hands.HandLandmark.INDEX_FINGER_DIP]
            chosenLandmarksText = 'Index, middle and ring finger DIP joints'
            chosenLandmarksID = 'DIP'
        else:
            chosenLandmarks = [mp_hands.HandLandmark.RING_FINGER_TIP,
                               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                               mp_hands.HandLandmark.INDEX_FINGER_TIP]
            chosenLandmarksText = 'Index, middle and ring finger tips'
            chosenLandmarksID = 'TIP'

    else:
        if AUTO_MODE:
            inp = str(sys.argv[4]).upper()[0]
        else:
            print('Track MCP joint, IP joint and tip of thumb, or track MCP,' +
                  ' PIP and DIP joints of\nindex finger?')
            inp = input('> Type in thumb or index: ').upper()[0]
        if inp == 'T':
            chosenLandmarks = [mp_hands.HandLandmark.THUMB_MCP,
                               mp_hands.HandLandmark.THUMB_IP,
                               mp_hands.HandLandmark.THUMB_TIP]
            chosenLandmarksText = 'Thumb tip, IP joint and MCP joint'
            chosenLandmarksID = 'thumb'
        else:
            chosenLandmarks = [mp_hands.HandLandmark.INDEX_FINGER_MCP,
                               mp_hands.HandLandmark.INDEX_FINGER_PIP,
                               mp_hands.HandLandmark.INDEX_FINGER_DIP]
            chosenLandmarksText = 'Index finger MCP, PIP and DIP joints'
            chosenLandmarksID = 'index'


def computeTremorPath():

    global capture

    minLeft = [videoWidth] * len(chosenLandmarks)
    maxRight = [0] * len(chosenLandmarks)
    path = [[] for i in range(len(chosenLandmarks))]
    pathFrameNumbers = []
    failedFrames = 0

    minLeftFrame = None
    maxRightFrame = None

    print('-' * 80)
    print('Computing tremor amplitude for %s...' % VIDEO_FILEPATH)

    # Configure mediapipe hand detector:
    detector = mp_hands.Hands(
        static_image_mode=False,  # False -> treat images as video sequence, to track hands between images. Allows faster and more accurate tracking.
        max_num_hands=1,  # Expect maximum of 1 hand in frame
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Model accuracy, 1 == more accurate but slower
    )

    frameN = 0
    while (True):
        readSuccess, frame = capture.read()
        if readSuccess:
            frameN += 1
            if START_FRAME <= frameN <= END_FRAME:
                if tremorType == Tremor.Postural:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detectionResults = detector.process(frameRGB)
                if not detectionResults.multi_hand_landmarks:
                    failedFrames += 1
                    continue
                landmarks = detectionResults.multi_hand_landmarks[0]

                # Draw hand landmarks onto frame:
                frameWithLandmarks = frame.copy()
                mp_drawing.draw_landmarks(
                    frameWithLandmarks,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                if GUI_HAND_TRACKING:
                    cv2.namedWindow('Hand Tracking')
                    cv2.imshow('Hand Tracking', frameWithLandmarks)
                    cv2.waitKey(1)

                # Array for storing x coordinates of finger pip joints:
                fingerLandmarkX = [None] * len(chosenLandmarks)

                for i in range(0, len(chosenLandmarks)):
                    fingerLandmarkX[i] = round(landmarks.landmark[
                        chosenLandmarks[i]].x * videoWidth)
                    if fingerLandmarkX[i] < minLeft[i]:
                        minLeft[i] = fingerLandmarkX[i]
                        if len(chosenLandmarks) == 1:
                            minLeftFrame = frameWithLandmarks
                        elif i == 1:
                            minLeftFrame = frameWithLandmarks
                    if fingerLandmarkX[i] > maxRight[i]:
                        maxRight[i] = fingerLandmarkX[i]
                        if len(chosenLandmarks) == 1:
                            maxRightFrame = frameWithLandmarks
                        elif i == 1:
                            maxRightFrame = frameWithLandmarks

                    path[i].append(fingerLandmarkX[i])

                pathFrameNumbers.append(frameN)

                print('Processing frame ' + str(frameN) + '/' + str(END_FRAME)
                      + '...', end='\r')
        else:
            print('\n', end='')
            if GUI_HAND_TRACKING:
                cv2.destroyAllWindows()
            break
    if not os.path.exists('data/key_frames'):
        os.makedirs('data/key_frames')
    cv2.imwrite('data/key_frames/' + videoFilename + '_' + chosenLandmarksID
                + '_leftMostFrame.jpg',
                minLeftFrame)
    cv2.imwrite('data/key_frames/' + videoFilename + '_' + chosenLandmarksID
                + '_rightMostFrame.jpg',
                maxRightFrame)

    capture.release()

    return path, pathFrameNumbers, failedFrames


# -----------------------------------------------------------------------------
# C A M E R A   P A R A M E T E R   C A L C U L A T I O N

def getDepthError(depth):
    if depth <= 45:
        return 0.115470054
    elif 45 < depth <= 55:
        return 0.081649658
    elif 55 < depth <= 65:
        return 0.141421356
    elif 65 < depth <= 75:
        return 0.203442594
    elif 75 < depth <= 85:
        return 0.324893145
    elif 85 < depth <= 95:
        return 0.37155828
    else:
        return 0.37859389


def calcPixelSize(imageResolution, sensorWidth, lensFocalLength, depth):
    """
    Calculate the width, in mm, of a pixel in an image at a given depth.

    Parameters
    ----------
    imageResolution : The width of an image in pixels.
    sensorWidth : The width of the camera sensor in mm.
    lensFocalLength : Focal length of the lens, in mm.
    depth : The depth at which the width of a pixel is to be calculated, in cm.

    Returns
    -------
    pixelSize : Size of a pixel at the given depth, in cm.
    pixelSizeErr : The error bounds of pixelSize, in +/- cm.
    """

    viewWidth = depth * sensorWidth / lensFocalLength
    viewWidthErr = (((getDepthError(depth) + depth) * sensorWidth /
                    lensFocalLength) - viewWidth)
    return viewWidth / imageResolution, viewWidthErr / imageResolution


def calcCropSensorWidth(sensorWidth, nativeAspectRatio, mediaAspectRatio):
    """
    Calculate effective/utilised width of camera sensor when image/video is
    recorded at non-native aspect ratio.
    """

    cropRatio = (nativeAspectRatio[1] / nativeAspectRatio[0]
                 ) / (mediaAspectRatio[1] / mediaAspectRatio[0])
    return sensorWidth * cropRatio


def calcSensorSize(focalLength, focalLength35mmEquiv, sensorAspectRatio):
    """
    Calculate the height and width of a camera sensor from its focal length,
    35mm equivalent focal length and aspect ratio.
    """

    cropFactor = focalLength35mmEquiv / focalLength
    # n.b. 43.27mm is the diagonal size of a full-frame 35mm sensor:
    sensorDiagonalLength = 43.27 / cropFactor
    angle = math.atan(sensorAspectRatio[1] / sensorAspectRatio[0])
    sensorWidth = math.cos(angle) * sensorDiagonalLength
    sensorHeight = math.sin(angle) * sensorDiagonalLength
    return sensorWidth, sensorHeight


# -----------------------------------------------------------------------------
# S E T U P   &   E T C   F U N C T I O N S

def loadConstantsFromConfigFile():
    """
    Load a configuration file and set the constants from the values in the
    file.
    """

    # n.b. see global definitions at the top of the file to see usage:
    global CAMERA_FOCAL_LENGTH, CAMERA_FOCAL_LENGTH_STD
    global CAMERA_NATIVE_ASPECT, CAMERA_VIDEO_ASPECT
    global START_FRAME, END_FRAME
    global GUI_HAND_TRACKING
    global AUTO_MODE
    global USE_CUSTOM_LANDMARKS
    global chosenLandmarks
    global SHOW_PLOT_LEGEND

    try:
        with open('D:\\Users\\LENOVO\\Desktop\\Development\\Capstone\\BioMechAnalysisOfParkinson\\tremor\\hta_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        if not ('CAMERA_FOCAL_LENGTH' in config.keys()
                and 'CAMERA_FOCAL_LENGTH_STD' in config.keys()
                and 'CAMERA_NATIVE_ASPECT' in config.keys()
                and 'CAMERA_VIDEO_ASPECT' in config.keys()
                and 'START_FRAME' in config.keys()
                and 'END_FRAME' in config.keys()
                and 'GUI_HAND_TRACKING' in config.keys()
                and 'AUTO_MODE' in config.keys()
                and 'USE_CUSTOM_LANDMARKS' in config.keys()
                and 'CUSTOM_LANDMARKS' in config.keys()
                and 'SHOW_PLOT_LEGEND' in config.keys()):
            raise Exception

        CAMERA_FOCAL_LENGTH = config.get('CAMERA_FOCAL_LENGTH')
        CAMERA_FOCAL_LENGTH_STD = config.get('CAMERA_FOCAL_LENGTH_STD')
        CAMERA_NATIVE_ASPECT = config.get('CAMERA_NATIVE_ASPECT')
        CAMERA_VIDEO_ASPECT = config.get('CAMERA_VIDEO_ASPECT')

        START_FRAME = config.get('START_FRAME')
        END_FRAME = config.get('END_FRAME')

        GUI_HAND_TRACKING = config.get('GUI_HAND_TRACKING')

        AUTO_MODE = config.get('AUTO_MODE')

        USE_CUSTOM_LANDMARKS = config.get('USE_CUSTOM_LANDMARKS')

        if USE_CUSTOM_LANDMARKS:
            chosenLandmarks = config.get('CUSTOM_LANDMARKS')

        SHOW_PLOT_LEGEND = config.get('SHOW_PLOT_LEGEND')

    except FileNotFoundError:
        print('ERROR: Could not find config file. There must be a config' +
              ' file with the name\nhta_config.yaml in the directory this' +
              ' is running from.')
        sys.exit()

    except Exception:
        print('ERROR: Config file was found, but does not contain the' +
              ' correct entries. Check\nthe config file and try again.')
        sys.exit()


def getVideoFilepath():
    global VIDEO_FILEPATH, videoFilename

    if AUTO_MODE:
        inp = str(sys.argv[1])
    else:
        inp = "C:\\Users\\Nitish\\OneDrive\\Pictures\\Camera Roll\\WIN_20230123_14_08_41_Pro.mp4"
        # inp = str(input('> Enter file path of video: '))

    VIDEO_FILEPATH = inp

    videoFilename = VIDEO_FILEPATH.split('/')[-1].split('.')[0]


def getDepthMeasurement():
    global HAND_DEPTH

    if AUTO_MODE:
        inp = sys.argv[2]
    else:
        # inp = input('> Enter depth measurement from TrueDepth camera, in cm: ')
        inp = 3
    HAND_DEPTH = float(inp)


def selectTremorType():
    global tremorType

    if AUTO_MODE:
        inp = str(sys.argv[3]).lower()[0]
    else:
        # inp = input('> Measure resting or postural tremor? (r/p): ').lower()[0]
        inp = 'r'
    if inp == 'r':
        tremorType = Tremor.Resting
    else:
        tremorType = Tremor.Postural


def openCaptureAndGetVideoInfo():
    global capture, videoWidth, videoFramerate

    capture = cv2.VideoCapture(VIDEO_FILEPATH)

    if tremorType == Tremor.Resting:
        videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Swap width and height, since postural videos are rotated 90 degrees:
        videoWidth = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoHeight = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    videoFramerate = round(capture.get(cv2.CAP_PROP_FPS))
    aspectRatio = videoWidth / videoHeight
    if aspectRatio != CAMERA_VIDEO_ASPECT[0] / CAMERA_VIDEO_ASPECT[1]:
        print('WARNING: Camera video aspect ratio may be wrong. ')


def printConfig():
    print('-' * 80)
    print('Tremor type to measure          : ' + tremorType.name)
    print('Hand landmarks to track         : ' + chosenLandmarksText)
    print('Video file path                 : ' + VIDEO_FILEPATH)
    resFpsStr = (str(videoWidth) + 'x' + str(int(videoWidth /
                 CAMERA_VIDEO_ASPECT[0] * CAMERA_VIDEO_ASPECT[1])) + ', '
                 + str(videoFramerate) + 'fps')
    print('Video resolution and frame rate : ' + resFpsStr)
    print('Video frames to analyse         : ' + str(START_FRAME) + ' to ' +
          str(END_FRAME))
    print('Camera focal length             : ' + str(CAMERA_FOCAL_LENGTH)
          + 'mm')
    print('Camera 35mm equiv. focal length : ' + str(CAMERA_FOCAL_LENGTH_STD)
          + 'mm')
    print('Camera aspect ratio             : ' + str(CAMERA_NATIVE_ASPECT[0])
          + ':' + str(CAMERA_NATIVE_ASPECT[1]))
    print('Video aspect ratio              : ' + str(CAMERA_VIDEO_ASPECT[0])
          + ':' + str(CAMERA_VIDEO_ASPECT[1]))
    print('Hand depth measurement          : ' + str(HAND_DEPTH) + 'cm')

    if not AUTO_MODE:
        # inp = input('> Check above values; ok to continue? (y/n): ').lower()[0]
        inp = 'y'
        if inp != 'y':
            sys.exit()


def writeOutResults(finalAmplitude, updrsRating, tremorPathRange,
                    amplitude2SdMax, totalError, amplitudeError,
                    pixelSizeError, trackingError, failedFrames):
    if not os.path.isfile('data/hta_results.csv'):
        with open('data/hta_results.csv', 'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            w.writerow(['Video File']
                       + ['Tremor Type']
                       + ['Hand Depth']
                       + ['Tracking Landmarks']
                       + ['UPDRS Severity Rating']
                       + ['Median Tremor Amplitude']
                       + ['Tremor Path Range']
                       + ['Maximum Tremor Amplitude 2 S.D. filtered']
                       + ['Total Error (+/-)']
                       + ['Error due to depth sensor inaccuracy (+/-)']
                       + ['Error due to pixel size discretion (+/-)']
                       + ['Error due to hand tracking inaccuracy (+/-)']
                       + ['Failed Frames'])

    with open('data/hta_results.csv', 'a', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow([videoFilename]
                   + [tremorType.name]
                   + [str('%.1f' % HAND_DEPTH)]
                   + [chosenLandmarksID]
                   + [str(updrsRating)]
                   + [str('%.2f' % finalAmplitude)]
                   + [str('%.2f' % tremorPathRange)]
                   + [str('%.2f' % amplitude2SdMax)]
                   + [str('%.2f' % totalError)]
                   + [str('%.2f' % amplitudeError)]
                   + [str('%.2f' % pixelSizeError)]
                   + [str('%.2f' % trackingError)]
                   + [failedFrames])
def writeResult(UDPS):
    return "ye le result" + str(UDPS)
# -----------------------------------------------------------------------------
# M A I N

def res():
    print('-' * 80)
    print('Hand Landmark -Capstone Stage 2')
    print('-' * 80)

    loadConstantsFromConfigFile()

    getVideoFilepath()

    getDepthMeasurement()

    # Select whether the input video depicts resting or postural tremor:
    selectTremorType()

    # Choose hand landmarks to track for tremor measurement:
    selectLandmarks()

    if AUTO_MODE:
        print('Automatic mode is on - parameters were obtained from command'
              + ' line arguments.')

    # Open video capture and get video info:
    openCaptureAndGetVideoInfo()

    # Print the configuration before starting video analysis:
    printConfig()

    # Compute the path of movement of the hand in the input video:
    path, pathTime, failedFrames = computeTremorPath()

    # Calculate camera sensor width from more readily-available camera
    # specifications:
    cameraSensorWidth, __ = calcSensorSize(CAMERA_FOCAL_LENGTH, CAMERA_FOCAL_LENGTH_STD,CAMERA_NATIVE_ASPECT)

    videoSensorWidth = calcCropSensorWidth(cameraSensorWidth,CAMERA_NATIVE_ASPECT,CAMERA_VIDEO_ASPECT)
    pixelSize, pixelSizeError = calcPixelSize(videoWidth,
                                              videoSensorWidth,
                                              CAMERA_FOCAL_LENGTH,
                                              HAND_DEPTH)
    amplitudePixelDistance = [None] * len(chosenLandmarks)
    amplitude2SdMaxPixelDistance = [None] * len(chosenLandmarks)
    tremorPathRangePixelDistance = [None] * len(chosenLandmarks)
    amplitude = [None] * len(chosenLandmarks)
    amplitude2SdMax = [None] * len(chosenLandmarks)
    tremorPathRange = [None] * len(chosenLandmarks)
    for i in range(0, len(chosenLandmarks)):
        # Calculate amplitude in pixels:
        amplitudePixelDistance[i], amplitude2SdMaxPixelDistance[i] = (
            calcPixelDistAmplitudeFromPath(path[i]))
        tremorPathRangePixelDistance[i] = max(path[i]) - min(path[i])
        # Convert amplitude from pixels to cm:
        amplitude[i] = amplitudePixelDistance[i] * pixelSize
        amplitude2SdMax[i] = amplitude2SdMaxPixelDistance[i] * pixelSize
        tremorPathRange[i] = tremorPathRangePixelDistance[i] * pixelSize

    finalAmplitude = sum(amplitude) / len(amplitude)

    amplitude2SdMaxAvg = sum(amplitude2SdMax) / len(amplitude2SdMax)

    tremorPathRangeAvg = sum(tremorPathRange) / len(tremorPathRange)

    amplitudeError = (finalAmplitude / pixelSize) * pixelSizeError
    pixelDiscretionError = (0.5 * pixelSize) * 2  

    # Calculate error due to inaccuracy in hand tracking:
    trackingError = calcErrorFromHandTracking(amplitude)

    totalError = amplitudeError + pixelDiscretionError + trackingError

    # Calculate an MDS-UPDRS tremor severity rating from the amplitude:
    updrsRating = calcUpdrsRating(finalAmplitude, totalError)
    UDPS = updrsRating
    # writeResult(updrsRating)
    if AUTO_MODE:
        writeOutResults(finalAmplitude, updrsRating, tremorPathRangeAvg,
                        amplitude2SdMaxAvg, totalError, amplitudeError,
                        pixelDiscretionError, trackingError, failedFrames)

    # Print results to console:
    # print('-' * 80)
    # print('Median tremor amplitude = %.2f +/- %.2f cm' %
    #       (finalAmplitude, totalError))
    # print('-' * 80)
    # print('UPDRS rating (from median tremor ampl.) : %s' % updrsRating)
    # UDPS = updrsRating
    # print('Max. tremor amplitude, 2 S.D. filtered  : %.2f cm'
    #       % amplitude2SdMaxAvg)
    # print('Tremor path range                       : %.2f cm'
    #       % tremorPathRangeAvg)
    # print('-' * 80)
    
    # print('-' * 80)
    # print('-' * 80)
    # print('Frames where hand detection failed: %d' % failedFrames)
    # print('-' * 80)
    # print("UDPRS",UDPS)
    data["UPDRS"] = UDPS
    return data

res()