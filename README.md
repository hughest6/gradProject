"# gradProject" 

reflectors.py contains a Reflector superclass with individual reflector type subclasses. These subclasses correspond to three different types of common scene reflectors for use in target scenes

scene.py contains the Scene class which generates an indiviudal scene and contains functions to add reflectors to the scene, generates information about the scene, and computes the total scene rcs.

rcs_stats.py contains helper functions for the scene class. These functions will generate meaningful statisics based on the scene radar cross section.