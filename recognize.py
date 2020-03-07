# USAGE
# python recognize.py --conf config/config.json

# import the necessary packages
from toolkit.utils import Config
from imutils.video import VideoStream
from imutils import paths
from threading import Thread
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime
from datetime import date
import RPi.GPIO as GPIO
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# set the Pi Traffic Light GPIO pins
red = 9
yellow = 10
green = 11

# setup the Pi Traffic Light
GPIO.setmode(GPIO.BCM)
GPIO.setup(red, GPIO.OUT)
GPIO.setup(yellow, GPIO.OUT)
GPIO.setup(green, GPIO.OUT)

def correct_passcode(p):
	# actuate a lock or (in our case) turn on the green light
	GPIO.output(green, True)

	# print status and play the correct sound file
	print("Welcome!")
	play_sound(p)

def incorrect_passcode(p):
	# turn on the red light
	GPIO.output(red, True)

	# print status and play the incorrect sound file
	print("Security breach!")
	play_sound(p)

def reset_lights():
	# turn off the lights
	GPIO.output(red, False)
	GPIO.output(yellow, False)
	GPIO.output(green, False)

def play_sound(p):
	# construct the command to play a sound, then execute the command
	command = "aplay {}".format(p)
	os.system(command)
	print(command)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
	help="path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the Twilio notifier
conf = Config(args["config"])

# grab the paths to gesture icon images and then initialize the icons
# dictionary where the key is the gesture name (derived from the image
# path) and the key is the actual icon image
print("Loading icons...")
imagePaths = paths.list_images(conf["assets_path"])
icons = {}

# loop over the image paths
for imagePath in imagePaths:
	# extract the gesture name (label) the icon represents from the
	# filename, load the icon, and then update the icons dictionary
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
	icon = cv2.imread(imagePath)
	icons[label] = icon

# grab the top-left and and bottom-right (x, y)-coordinates for the
# gesture capture area
TOP_LEFT = tuple(conf["top_left"])
BOT_RIGHT = tuple(conf["bot_right"])

# load the trained gesture recognizer model and the label binarizer
print("Loading model...")
model = load_model(str(conf["model_path"]))
lb = pickle.loads(open(str(conf["lb_path"]), "rb").read())

# start the video stream thread
print("Starting video stream thread...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# initialize the current gesture, a bookkeeping variable used to keep
# track of the number of consecutive frames a given gesture has been
# classified as
currentGesture = [None, 0]

# initialize the list of input gestures recognized from the user
# along with the timestamp of when all four gestures were entered
gestures = []
enteredTS = None

# initialize two booleans used to indicate (1) whether or not the
# alarm has been raised and (2) if the correct pass code was entered
alarm = False
correct = False

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream and grab the
	# current timestamp
	frame = vs.read()
	timestamp = datetime.now()

	# resize the frame and then flip it horizontally
	frame = imutils.resize(frame, width=500)
	frame = cv2.flip(frame, 1)

	# clone the original frame and then draw the gesture capture area
	clone = frame.copy()
	cv2.rectangle(clone, TOP_LEFT, BOT_RIGHT, (0, 0, 255), 2)

	# only perform hand gesture classification if the current gestures
	# list is not already full
	if len(gestures) < 4:
		# extract the hand gesture capture ROI from the frame, convert
		# the ROI to grayscale, and then threshold it to reveal a
		# binary mask of the hand
		roi = frame[TOP_LEFT[1]:BOT_RIGHT[1], TOP_LEFT[0]:BOT_RIGHT[0]]
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi = cv2.threshold(roi, 75, 255, cv2.THRESH_BINARY)[1]
		visROI = roi.copy()

		# now that we have the hand region we need to resize it to be
		# the same dimensions as what our model was trained on, scale
		# it to the range [0, 1], and prepare it for classification
		roi = cv2.resize(roi, (64, 64))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)

		# classify the input image
		proba = model.predict(roi)[0]
		label = lb.classes_[proba.argmax()]

		# check to see if the label from our model matches the label
		# from the previous classification
		if label == currentGesture[0] and label != "ignore":
			# increment the current gesture count
			currentGesture[1] += 1

			# check to see if the current gesture has been recognized
			# for the past N consecutive frames
			if currentGesture[1] == conf["consec_frames"]:
				# update the gestures list with the predicted label
				# and then reset the gesture counter
				gestures.append(label)
				currentGesture = [None, 0]

		# otherwise, reset the current gesture count
		else:
			currentGesture = [label, 0]

	# turn on the yellow light if we have at least 1 gesture
	if len(gestures) >= 1 and len(gestures) < 4:
		GPIO.output(yellow, True)

	# otherwise, turn the yellow light off
	else:
		GPIO.output(yellow, False)

	# initialize the canvas used to draw recognized gestures
	canvas = np.zeros((250, 425, 3), dtype="uint8")

	# loop over the number of hand gesture input keys
	for i in range(0, 4):
		# compute the starting x-coordinate of the entered gesture
		x = (i * 100) + 25

		# check to see if an input gesture exists for this index, in
		# which case we should display an icon
		if len(gestures) > i:
			canvas[65:165, x:x + 75] = icons[gestures[i]]

		# otherwise, there has not been an input gesture for this icon
		else:
			# draw a white box on the canvas, indicating that a
			# gesture has not been entered yet
			cv2.rectangle(canvas, (x, 65), (x + 75, 165),
				(255, 255, 255), -1)

	# initialize the status as "waiting" (implying that we're waiting
	# for the user to input four gestures) along with the color of the
	# status text
	status = "Waiting"
	color = (255, 255, 255)

	# check to see if there are four gestures in the list, implying
	# that we need to check the pass code
	if len(gestures) == 4:
		# if the timestamp of when the four gestures has been entered
		# has not been initialized, initialize it
		if enteredTS is None:
			enteredTS = timestamp

		# initialize our status, color, and sound path for the
		# "correct" pass code
		status = "Correct"
		color = (0, 255, 0)
		audioPath = conf["correct_audio"]

		# check to see if the input gesture pass code is correct
		if gestures == conf["passcode"]:
			# if we have not taken action for a correct pass code,
			# take the action
			if not correct:
				t = Thread(target=correct_passcode, args=(audioPath,))
				t.daemon = True
				t.start()
				correct = True

		# otherwise, the pass code is incorrect
		else:
			# update the status, color and audio path
			status = "Incorrect"
			color = (0, 0, 255)
			audioPath = conf["incorrect_audio"]

			# if the alarm has not already been raised, raise it
			if not alarm:
				t = Thread(target=incorrect_passcode,
					args=(audioPath,))
				t.daemon = True
				t.start()
				alarm = True

		# after a correct/incorrect pass code we will show the status
		# for N seconds
		if (timestamp - enteredTS).seconds > conf["num_seconds"]:
			# reset the gestures list, timestamp, and alarm/correct
			# booleans
			gestures = []
			enteredTS = None
			alarm = False
			correct = False
			reset_lights()

	# draw the timestamp and status on the canvas
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	status = "Status: {}".format(status)
	cv2.putText(canvas, ts, (10, canvas.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, color, 2)

	# show ROI we're monitoring, the output frame, and passcode info
	cv2.imshow("ROI", visROI)
	cv2.imshow("Security Feed", clone)
	cv2.imshow("Passcode", canvas)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		reset_lights()
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
