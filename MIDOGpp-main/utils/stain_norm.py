from tiatoolbox.tools import stainnorm
from tiatoolbox.tools.stainextract import VahadaneExtractor
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils import imread
from PIL import Image
import numpy as np

# Define the target image used (it must contains the necessary color spectrum) 
target_image_path = "/images/009.tiff"

# Define the extractor and get the stain_matrix
extractor = VahadaneExtractor()
img = imread(target_image_path)
stain_matrix = extractor.get_stain_matrix(img)

# Get the stain metrix
def get_target_stain_matrix():
  
  return stain_matrix

# Stain normalization established for this work ('Vahadane')
def stain_norm_established(source_image):

  new_normalized_source = normalizer_work.transform(source_image)
  
  return new_normalized_source
  
# Stain normalization using Vahadane
def stain_norm_Vahadane(source_image):
  normalizer = get_normalizer('Vahadane')
  normalizer.fit(target_image)
  new_normalized_source = normalizer.transform(source_image)
  
  return new_normalized_source
  
  
# Stain normalization using Macenko
def stain_norm_Macenko(source_image):
  normalizer = get_normalizer('Macenko')
  normalizer.fit(target_image)
  new_normalized_source = normalizer.transform(source_image)
  
  return new_normalized_source
  

# Stain normalization using Reinhard
def stain_norm_Reinhard(source_image):
  normalizer = get_normalizer('Reinhard')
  normalizer.fit(target_image)
  new_normalized_source = normalizer.transform(source_image)
  
  return new_normalized_source
  
  
# Stain normalization using Ruifrok
def stain_norm_Ruifrok(source_image):
  normalizer = get_normalizer('Ruifrok')
  normalizer.fit(target_image)
  new_normalized_source = normalizer.transform(source_image)
  
  return new_normalized_source
  
  
