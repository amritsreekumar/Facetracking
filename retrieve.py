import json

def retrieve_by_ID(ID):
	f = open('metadata.json',)
	data = json.load(f)
	data = json.loads(data)
	for path, detail in data.items():
		print(detail)

retrieve_by_ID(0)

