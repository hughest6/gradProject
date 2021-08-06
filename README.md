"# gradProject" 

reflectors.py contains a Reflector superclass with individual reflector type subclasses. These subclasses correspond to three different types of common scene reflectors for use in target scenes

scene.py contains the Scene class which generates an indiviudal scene and contains functions to add reflectors to the scene, generates information about the scene, and computes the total scene rcs.

rcs_stats.py contains helper functions for the scene class. These functions will generate meaningful statisics based on the scene radar cross section.

data_processing.py generates a final output .csv file that contains the rcs statistics for a given number of scenes. This can then be used to train ML models.

classification.py contains functions necessary for training and testing machine learning models.

main.py contains all of the code to get the scenes generated and the machine learning models generated.
