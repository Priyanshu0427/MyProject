import os
import cv2
import cvlib
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
import pytesseract
import pdf2image
from pytesseract import Output

eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

# pytesseract.pytesseract.tesseract_cmd =r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# poppler_path=r'C:\\poppler\\poppler-21.03.0\\Library\\bin'

cfg_file_path='yolov3_last_training.cfg'
weights_file_path='yolov3_training_last.weights'
names_file_path = 'names.names'

# word_list = ['what', 'want', 'You', 'you', 'know', "touch", 'those', 'Data', 'data']        # list of words to be masked

def nudity_blur(img, cfg_file, weight_file, name_file):
	'''returns the censored image, label for the part and confidence for that part'''
	classes=[]
	with open(name_file, 'r') as f:
		classes=[line.strip() for line in f.readlines()]
    # give configuration and weight file
	net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file) 
	height, width, channels = img.shape
    # convert image to blob
	blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0),swapRB=True, crop=False )
    # feeding blob input to yolo input
	net.setInput(blob)
    # getting last layer
	last_layer = net.getUnconnectedOutLayersNames()
    # getting output from this layer
	last_out = net.forward(last_layer)
    
	boxes=[]         # for storing coordinates of rectangle
	confidences=[]   # for storing probabilities
	class_ids=[]     # for storing the label index
    
    
	for output in last_out:
		for detection in output:
			score = detection[4:]                 # probabilities are after 5th element first 4 are cooordinates
			class_id = np.argmax(score)           # gives index of highest probability
			confidence = score[class_id]          # gives the highest probability
			if confidence > 0.05:                  # if the probability of happening is above 20%
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				x = int(center_x - w/2)
				y = int(center_y - h/2)

				boxes.append([x,y,w,h])
				confidences.append((float(confidence)))
				class_ids.append(class_id)
                
	labels=[]
	conf=[]
	indices= np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.05,0.4))
#     indices = np.array(index)
	for i in indices.flatten():
		x,y,w,h = boxes[i]                              # returns coordinates
		label = str(classes[class_ids[i]])              # returns label for each image
		confidence = str(round(confidences[i],2))       # returns confidence for each label
        # make blur
      
        #roi = img[y:y+h, x:x+w]
		img[y:y+h, x:x+w]=cv2.medianBlur(img[y:y+h, x:x+w], ksize=201)
		labels.append(label)
		conf.append(confidence)
    
    
	return(img, labels, conf)

# def remove_punc(word):
#     punc = [',', '.', '-', '/', '@', '"']
#     for ele in word:  
#         if ele in punc:  
#             word = word.replace(ele, "") 
#     return word

# def word_coor(image):
#     boxes=[]
#     texts=[]
#     rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # converts image to rgb
#     results = pytesseract.image_to_data(rgb, output_type=Output.DICT)
#     if set(results['text']) == '':
#         return ('No text detected!!!')
#     else:
#         for i in range(0,len(results['top'])):  ## iterating through all the values for the words present
#             text = remove_punc(results['text'][i])          ## returning the text value read by OCR
#             conf = int(results['conf'][i])          ## returning confidence of word read by OCR
            
#             x= results['left'][i]              ## top left x coordinates of word read
#             y= results['top'][i]               ## top left y coordinate of word read
#             w= results['width'][i]             ## width of word read
#             h= results['height'][i]            ## height of word read
            
#             if conf > 50:
#                 boxes.append([x,y,w,h])
#                 texts.append(text) 
#         return list(zip(texts, boxes))
        
# def word_mask(img, list):
#     for word, coor in word_coor(img):
#         if word in list:
#             x,y,w,h = coor
#             roi = img[y:y+h, x:w+x]
#             img[y:y+h, x:w+x] = cv2.medianBlur(roi, ksize=151)
#     return img

def face_blur(img):
	coor, _ = cvlib.detect_face(img)
	for face in coor:
		x,y,w,h = face
		roi = img[y:h, x:w]
		img[y:h, x:w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.3,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.6,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.6,8)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.5,10)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

# def pdf_2_images(pdf_path, pdf_dest, file_name):
#     images = pdf2image.convert_from_path(pdf_path = pdf_path, poppler_path=poppler_path)
#     img_list=[]
#     for i in range(len(images)):
#         img = np.array(images[i])
#         image = word_mask(img, word_list)
# #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         Img = Image.fromarray(image)
#         img_list.append(Img)
#     return img_list[0].save(os.path.join(pdf_dest,(file_name+'.pdf')), save_all=True, append_images=img_list)

# def file_selector(folder_path=os.getcwd()):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)


def about():
	st.markdown('''This Web App is made for censoring(blurring) the NSWF(Not Safe For Work) materials, that includes personal pictures depicting Nudity.
		This App will blur the inappropriate areas and body. Along with that we have provided option for blurrring Eyes, Smile, Face and Nudity.''')

	st.markdown('''YOLO custom object dedection is used for detection of Nudity whereas HAAR Cascade Classifier is used for detecting Face, Smile and Eyes.''')


def main():
	st.title("Object Detection and Masking")
	st.subheader('For Recorded as well as Real-time media')
	st.write('Using YOLOV3 object detection and Haar Cascade Classifier we detect the NSWF parts and blur them with OpenCV')

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox('Select an option', activities)

	if choice == 'Home':
		st.write('Go to the about section to know more about it')

		file_type = ['Image', 'Video']
		file_choice = st.sidebar.radio('Select file type', file_type)

		if file_choice == 'Video':
			file = st.file_uploader('Choose file', ['mp4'])

			if file is not None:

				tfile = tempfile.NamedTemporaryFile(delete=False)
				tfile.write(file.read())


				choice_type = st.sidebar.radio('Make your choice', ['Original', 'Eyes', 'Face', 'Smile', 'Nudity'])

				if st.button('Process'):
					if choice_type == 'Original':
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
	    			# if frame is read correctly ret is True
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							# if frame == None:
							# 	pass
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(frame)

					elif choice_type == 'Eyes':
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
	    			# if frame is read correctly ret is True
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							# if frame == None:
							# 	pass
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(blur_eyes_video(frame))


					elif choice_type == 'Face':
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
	    			# if frame is read correctly ret is True
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							# if frame == None:
							# 	pass
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(face_blur(frame))


					elif choice_type == 'Smile':
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
	    			# if frame is read correctly ret is True
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							# if frame == None:
							# 	pass
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(blur_smile_video(frame))



					elif choice_type == 'Nudity':
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
	    			# if frame is read correctly ret is True
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							# if frame == None:
							# 	pass
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(image)
							if len(label) == 0:
								st.info('No Nudity present in the video')

					# elif choice_type == 'Text':
					# 	vf = cv2.VideoCapture(tfile.name)
					# 	stframe = st.empty()

					# 	while vf.isOpened():
					# 		ret, frame = vf.read()
	    # 			# if frame is read correctly ret is True
					# 		if not ret:
					# 			print("Can't receive frame (stream end?). Exiting ...")
					# 			break
					# 		# if frame == None:
					# 		# 	pass
					# 		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					# 		stframe.image(word_mask(frame, word_list))

			else:
				st.warning('Upload the file first')



		elif file_choice == 'Image':

			file=st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'webp'])

			if file is not None:
				
				image = Image.open(file)
				image = np.array(image)

				choice_type = st.sidebar.radio('Make a choice', ('Original','Eyes', 'Face', 'Smile', 'Nudity'))

				if st.button('Process'):
					if choice_type == 'Original':
						result_image = image
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} image returned')

						
					elif choice_type == 'Eyes':
						result_image= blur_eyes(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if blur_eyes(image) == 'No Eyes Detected':
							st.info('No Eyes Detected')
						# st.info(blur_eyes(image))	
					elif choice_type == 'Face':
						result_image= face_blur(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if face_blur(image) == 'No Face Detected':
							st.info('No Face Detected')
						# st.info(blur_face(image))
					elif choice_type == 'Smile':
						result_image= blur_smile(image)
						st.image(result_image, use_column_width=True)
						st.info(f'{choice_type} of the image got blurred')
						if blur_smile(image) == 'No Smile Detected':
							st.info('No Smile Detected')
						# st.info(blur_smile(image))
					elif choice_type == 'Nudity':
						result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
						st.image(result_image, use_column_width=True)
						if len(confidence) ==0:
							st.info('No nudity present in the image.')
						else:
							x=round(np.mean([float(i) for i in confidence])*100,2)
							st.info(f'Nudity percentage: {x}%')

				# 		elif choice_type == 'Text':

				# 			# word = ['keep', 'You']      # list of words to be masked
				# 			result_image = word_mask(image, word_list)
				# 			st.image(result_image, use_column_width=True)
				# else:
				# 	filename = file_selector()
				# 	pdf_2_images(pdf_path=filename, pdf_dest=os.getcwd(), file_name='masked_pdf')
				# 	st.info('pdf masked successfully')



					


	elif choice =='About':
		about()

if __name__ == '__main__':
	main()

