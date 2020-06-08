import json
import cv2

def retrieve_by_ID(ID):
	f = open('metadata.json',)
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

def retrieve_by_name(name):
	f = open('metadata.json',)
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


def retrieve_by_date(date):
	f = open('metadata.json',)
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


def retrieve_by_datetime(date, time):
	f = open('metadata.json',)
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

#retrieve_by_ID('0')
#retrieve_by_name('Amrit')
#retrieve_by_date('08:06:2020')
retrieve_by_datetime('08:06:2020','12:20')


