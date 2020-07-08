# Facetracking
Tracking and identifying multiple faces realtime

Run the facetrack.py file in python3 to locate faces and identify the names or its corresponding ID's.

Each frame is saved to an "outputframes" folder and its metadata that contains the name or the ID of the correspodning face(s) identified in the frame. 

"Classifier.pkl" is the pickle file that classifies each face into the corresponding person's name. It is set to a certain threshold set using the probability based on the eucledian distance between the two face vectors. 
This threshold could be changed inside the "facetrack.py" file as per requirement.

While the faces are being detected, the face embedding are stored in the "embeddings.json" file. This file is read continuously as each frame passes so as to detect the trajectory of each face and give the same ID for each passing frame.


The "retrieve.py" file can be used to easily track the faces as per the requirements. This file contains the code to retrieve the faces using four different parameters:

(The source variable is the camera number of which we need to retrieve the visuals from. Each camera is named from 1 to n inside the "facetrack.py" and should have separate "facetrack.py" files and its dependencies but the same "embeddings.json".)

	retrieve_by_ID(source, ID):
		This function retrieves the metadata of the frames corresponding to the ID we give and plays it in chronological order.
	
	retrieve_by_name(source, name):
		Retrieves the metadata using the source and the name of the required face as parameters.

	retrieve_by_date(source, date):
		Plays the frames in chronological order from the given start date as parameter.

	retrieve_by_datetime(source, date, time):
		Plays the frames in chronological order from the given start date and specific time as parameter.

