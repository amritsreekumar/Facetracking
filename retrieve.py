import json
import cv2

def retrieve_by_ID(source, ID):
	outputframes = 'outputframes' + str(source) + '/'
	f = open(outputframes + 'metadata.json',)
	data = json.load(f)
	data = json.loads(data)
	for path, detail in data.items():
		for key, value in detail[0].items():
			if key == ID:
				path = path + '.png'
				print(path)
				frame = cv2.imread(path)
				cv2.imshow('Frame', frame)
				if cv2.waitKey(200) & 0xFF == ord('q'):
					break

def retrieve_by_name(source, name):
	outputframes = 'outputframes' + str(source) + '/'
	f = open(outputframes + 'metadata.json',)
	data = json.load(f)
	data = json.loads(data)
	for path, detail in data.items():
		for key, value in detail[0].items():
			if value == name:
				path = path + '.png'
				print(path)
				frame = cv2.imread(path)
				cv2.imshow('Frame', frame)
				if cv2.waitKey(200) & 0xFF == ord('q'):
					break


def retrieve_by_date(source, date):
	outputframes = 'outputframes' + str(source) + '/'
	f = open(outputframes + 'metadata.json',)
	data = json.load(f)
	data = json.loads(data)
	for path, detail in data.items():
		if detail[1].startswith(date):
			path = path + '.png'
			print(path)
			frame = cv2.imread(path)
			cv2.imshow('Frame', frame)
			if cv2.waitKey(200) & 0xFF == ord('q'):
				break


def retrieve_by_datetime(source, date, time):
	outputframes = 'outputframes' + str(source) + '/'
	f = open(outputframes + 'metadata.json',)
	data = json.load(f)
	data = json.loads(data)
	datetime = date + ';' + time
	for path, detail in data.items():
		if detail[1].startswith(datetime):
			path = path + '.png'
			print(path)
			frame = cv2.imread(path)
			cv2.imshow('Frame', frame)
			if cv2.waitKey(200) & 0xFF == ord('q'):
				break

#retrieve_by_ID(0,'0')
retrieve_by_name(0,'Amrit')
#retrieve_by_date(0,'08:06:2020')
#retrieve_by_datetime(0,'08:06:2020','12:20')


