
# ==============================================================================
''' Paths to data, labels and model weights/structure
Note that the tfrecs/pickles have to be made and put in the correct folder using adapations of the resource generation scrips.

The resources folder also has to be made manually and has to contain the weights from the public yamnet repo.
'''
# ==============================================================================
paths = {}

paths['fsd_train_labels_path'] = 'fsd-tfrecs/train_post_competition.csv'
paths['fsd_test_labels_path'] = 'fsd-tfrecs/test_post_competition_scoring_clips.csv'

##############################################
# Part related to private data removed here. #
# Code might not run/run as intended.	     #
##############################################

paths['fsd_tfrec_train'] = 'fsd-tfrecs/train.tfrec'
paths['fsd_tfrec_test'] = 'fsd-tfrecs/test.tfrec'

##############################################
# Part related to private data removed here. #
# Code might not run/run as intended.	     #
##############################################

paths['yamnet_weights'] = 'resources/yamnet.h5'
paths['yamnet_classes'] = 'resources/yamnet_class_map.csv'
paths['yamnet_embeddings'] = 'resources/yamnet_embeddings.pickle'

