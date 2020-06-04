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


retrieve_by_ID('0')

