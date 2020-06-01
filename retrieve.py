import json

def retrieve_by_ID(ID):
	metadata = json.load(open("metadata.json"))
	for path, detail in metadata.items():
		print(detail)

retrieve_by_ID(0)

