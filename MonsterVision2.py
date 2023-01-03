#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import apriltag
import depthai as dai
import numpy as np
import math
import time
from networktables import NetworkTables
from networktables import NetworkTablesInstance
cscoreAvailable = True
try:
    from cscore import CameraServer
except ImportError:
    cscoreAvailable = False

import json
# import socket

ROMI_FILE = "/boot/romi.json"
FRC_FILE = "/boot/frc.json"
NN_FILE = "/boot/nn.json"
CAMERA_FPS = 50
DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....
PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200

INCHES_PER_MILLIMETER = 39.37 / 1000
bbfraction = 0.2

openvinoVersions = dai.OpenVINO.getVersions()
openvinoVersionMap = {}
for v in openvinoVersions:
    openvinoVersionMap[dai.OpenVINO.getVersionName(v)] = v


# Return True if we're running on Romi.  False if we're a coprocessor on a big 'bot

def is_romi():
    try:
        with open(ROMI_FILE, "rt", encoding="utf-8") as f:
            json.load(f)
            # j = json.load(f)
    except OSError as err:
        print("Could not open '{}': {}".format(ROMI_FILE, err), file=sys.stderr)
        return False
    return True


def is_frc():
    try:
        with open(FRC_FILE, "rt", encoding="utf-8") as f:
            json.load(f)
    except OSError as err:
        print("Could not open '{}': {}".format(FRC_FILE, err), file=sys.stderr)
        return False
    return True



# hasDisplay = not is_romi() and not is_frc()
hasDisplay = False

def read_nn_config():
    try:
        with open(NN_FILE, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(NN_FILE, err), file=sys.stderr)
        return {}

    # top level must be an object
    if not isinstance(j, dict):
        parse_error("must be JSON object")
        return {}

    return j

# Required information for calculating spatial coordinates on the host
monoHFOV = np.deg2rad(73.5)
tanHalfHFOV = math.tan(monoHFOV / 2.0)

# Helper function for calc_spatials
def calc_angle(offset, depthWidth):
    return math.atan(math.tan(monoHFOV / 2.0) * offset / (depthWidth / 2.0))

def calc_tan_angle(offset, depthWidth):
    return offset * tanHalfHFOV / (depthWidth / 2.0)

# This code assumes depth symmetry around the centroid

# Calculate spatial coordinates from depth map and bounding box (ROI)
def calc_spatials(bbox, centroidX, centroidY, depth, depthWidth, averaging_method=np.mean):
    xmin, ymin, xmax, ymax = bbox
    # Decrese the ROI to 1/3 of the original ROI
    deltaX = int((xmax - xmin) * 0.33)
    deltaY = int((ymax - ymin) * 0.33)
    xmin += deltaX
    ymin += deltaY
    xmax -= deltaX
    ymax -= deltaY
    if xmin > xmax:  # bbox flipped
        xmin, xmax = xmax, xmin
    if ymin > ymax:  # bbox flipped
        ymin, ymax = ymax, ymin

    if xmin == xmax or ymin == ymax: # Box of size zero
        return None

    # Calculate the average depth in the ROI.
    depthROI = depth[ymin:ymax, xmin:xmax]
    averageDepth = averaging_method(depthROI)

    # mid = int(depth.shape[0] / 2) # middle of the depth img
    bb_x_pos = centroidX - int(depth.shape[1] / 2)
    bb_y_pos = centroidY - int(depth.shape[0] / 2)

    # angle_x = calc_angle(bb_x_pos, depthWidth)
    # angle_y = calc_angle(bb_y_pos, depthWidth)
    tanAngle_x = calc_tan_angle(bb_x_pos, depthWidth)
    tanAngle_y = calc_tan_angle(bb_y_pos, depthWidth)

    z = averageDepth
    x = z * tanAngle_x
    y = -z * tanAngle_y

    return (x,y,z)


def average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = (pt2[0] - pt1[0]) * factor / 2
    y_shift = (pt2[1] - pt1[1]) * factor / 2
    av_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    av_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return av_pt1, av_pt2


def parse_error(mess):
    """Report parse error."""
    print("config error in '" + FRC_FILE + "': " + mess, file=sys.stderr)


def read_frc_config():
    global team
    global server
    global hasDisplay

    try:
        with open(FRC_FILE, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(FRC_FILE, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parse_error("must be JSON object")
        return False

    # Is there an desktop display?
    try:
        hasDisplay = j["hasDisplay"]
    except KeyError:
        hasDisplay = False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parse_error("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        s = j["ntmode"]
        if s.lower() == "client":
            server = False
        elif s.lower() == "server":
            server = True
        else:
            parse_error("could not understand ntmode value '{}'".format(s))

    return True


read_frc_config()
nnJSON = read_nn_config()
LABELS = nnJSON['mappings']['labels']
nnConfig = nnJSON['nn_config']

ntinst = NetworkTablesInstance.getDefault()

if server:
    print("Setting up NetworkTables server")
    ntinst.startServer()
else:
    print("Setting up NetworkTables client for team {}".format(team))
    ntinst.startClientTeam(team)
    ntinst.startDSClient()

sd = NetworkTables.getTable("MonsterVision")

if cscoreAvailable:
    cs = CameraServer.getInstance()
    cs.enableLogging()
    output = cs.putVideo("MonsterVision", PREVIEW_WIDTH, PREVIEW_HEIGHT) # TODOnot

sd.putString("ObjectTracker", "Hi Ritchie")
'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''

# Get path to blob

blob = nnConfig['blob']
nnBlobPath = str((Path(__file__).parent / Path('models/' + blob)).resolve().absolute())

if not Path(nnBlobPath).exists():
    import sys

    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = LABELS

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

try:
    openvinoVersion = nnConfig['openvino_version']
except KeyError:
    openvinoVersion = ''

if openvinoVersion != '':
    pipeline.setOpenVINOVersion(openvinoVersionMap[openvinoVersion])

try:
    inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))
except KeyError:
    inputSize = (300, 300)

family = nnConfig['NN_family']
if family == 'mobilenet':
    detectionNodeType = dai.node.MobileNetSpatialDetectionNetwork
elif family == 'YOLO':
    detectionNodeType = dai.node.YoloSpatialDetectionNetwork
else:
    raise Exception(f'Unknown NN_family: {family}')

try:
    bbfraction = nnConfig['bb_fraction']
except KeyError:
    bbfraction = bbfraction			# No change fromn default



# Create the spatial detection network node - either MobileNet or YOLO (from above)

spatialDetectionNetwork = pipeline.create(detectionNodeType)

# Set the NN-specific stuff

if family == 'YOLO':
    spatialDetectionNetwork.setNumClasses(nnConfig['NN_specific_metadata']['classes'])
    spatialDetectionNetwork.setCoordinateSize(nnConfig['NN_specific_metadata']['coordinates'])
    spatialDetectionNetwork.setAnchors(nnConfig['NN_specific_metadata']['anchors'])
    spatialDetectionNetwork.setAnchorMasks(nnConfig['NN_specific_metadata']['anchor_masks'])
    spatialDetectionNetwork.setIouThreshold(nnConfig['NN_specific_metadata']['iou_threshold'])
    spatialDetectionNetwork.setConfidenceThreshold(nnConfig['NN_specific_metadata']['confidence_threshold'])
else:
    x = nnConfig['confidence_threshold']
    spatialDetectionNetwork.setConfidenceThreshold(x)

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(inputSize)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(CAMERA_FPS)
print("Camera FPS: {}".format(camRgb.getFps()))

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(inputSize[0], inputSize[1])

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(bbfraction)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    frame_counter = 0

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        counter += 1
        current_time = time.monotonic()
        if (current_time - startTime) > 1:
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_RAINBOW)

        detections = inDet.detections
        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

#                print(xmin, ymin, xmax, ymax)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width = frame.shape[1]

        # re-initializes objects to zero/empty before each frame is read
        objects = []
        s_detections = sorted(detections, key=lambda det: det.label * 100000 + det.spatialCoordinates.z)

        for detection in s_detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = labelMap[detection.label]

            except KeyError:
                label = detection.label

            # Draw the BB over which the depth is computed
            avg_pt1, avg_pt2 = average_depth_coord([detection.xmin, detection.ymin],
                                                   [detection.xmax, detection.ymax],
                                                   bbfraction)
            avg_pt1 = int(avg_pt1[0] * width), int(avg_pt1[1] * height)
            avg_pt2 = int(avg_pt2[0] * width), int(avg_pt2[1] * height)

            cv2.rectangle(frame, avg_pt1, avg_pt2, (0, 255, 255), 1)
            # Choose the color based on the label

            if detection.label == 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            #print(detection.spatialCoordinates.x, detection.spatialCoordinates.y, detection.spatialCoordinates.z)

            x = round(int(detection.spatialCoordinates.x * INCHES_PER_MILLIMETER), 1)
            y = round(int(detection.spatialCoordinates.y * INCHES_PER_MILLIMETER), 1)
            z = round(int(detection.spatialCoordinates.z * INCHES_PER_MILLIMETER), 1)

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"X: {x} in", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {y} in", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {z} in", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            objects.append({"objectLabel": LABELS[detection.label], "x": x,
                            "y": y, "z": z,
                            "confidence": round(detection.confidence, 2)})

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 255, 255))

        # Look for apriltag(s)

        options = apriltag.DetectorOptions(families="tag16h5")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)

        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            res = calc_spatials((ptA[0], ptA[1], ptC[0], ptC[1]), cX, cY, depthFrame, inputSize[0])
            if res == None:
                continue
            (atX, atY, atZ) = res
            atX = round((atX * INCHES_PER_MILLIMETER), 1)
            atY = round((atY * INCHES_PER_MILLIMETER), 1)
            atZ = round((atZ * INCHES_PER_MILLIMETER), 1)
            # draw the bounding box of the AprilTag detection
            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            wd = abs(ptC[0] - ptA[0])
            ht = abs(ptC[1] - ptA[1])
            lblX = int(cX - wd/2)
            lblY = int(cY - ht/2)
            # draw the tag family on the image
            tagID= '{}: {}'.format(r.tag_family.decode("utf-8"), r.tag_id)
            cv2.putText(frame, tagID, (lblX, lblY - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
            cv2.putText(frame, f"X: {atX} in", (lblX, lblY - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
            cv2.putText(frame, f"Y: {atY} in", (lblX, lblY - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
            cv2.putText(frame, f"Z: {atZ} in", (lblX, lblY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))

            objects.append({"objectLabel": tagID, "x": atX, "y": atY, "z": atZ})

        if hasDisplay:
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("preview", frame)

        # Take our list of objects found and dump it to JSON format.  Then write the JSON string to the
        # ObjectTracker key in the Network Tables

        jsonObjects = json.dumps(objects)
        sd.putString("ObjectTracker", jsonObjects)
        ntinst.flush()

        # Display the Frame

        if cscoreAvailable:
            if frame_counter % (CAMERA_FPS / DESIRED_FPS) == 0:
                output.putFrame(frame)

            frame_counter += 1

        if hasDisplay and cv2.waitKey(1) == ord('q'):
            break
