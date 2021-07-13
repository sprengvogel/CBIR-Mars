# Base training parameters
LR = 0.0001 #0.0001
BATCH_SIZE = 256
EPOCHS = 30
DENSENET_NUM_FEATURES = 1024
ADAM_BETAS = (0.9,0.99) #(0.5,0.7)

# Triplet distance margin
MARGIN = 0.2 #0.2

# Hashin parameters
HASH_BITS = 128
LAMBDA1 = 0.001
LAMBDA2 = 1

# Densenet type selection
DENSENET_TYPE = "imagenet"
#DENSENET_TYPE = "domars16k_classifier"
#DENSENET_TYPE = "domars16k_triplet"

# Interclass triplets parameters
INTERCLASSTRIPLETS = False
KMEANS_CLUSTERS = 30

# Domain whitening parameters
DOMAIN_ADAPTION = False
DA_GROUP_SIZE = 32

# Multiview parameters
MULTIVIEWS= False
PROJ_DIM = 128
TEMPERATURE = 0.1
