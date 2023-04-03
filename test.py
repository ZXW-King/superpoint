import glob
import os
basedir = '/media/xin/data1/shujuji0310/小广场/高光/recordgcgaoguang500/imsee_data.bag.imgs.L/'
img_glob = "*.png"
search = os.path.join(basedir, img_glob)
listing = glob.glob(search)
listing.sort()
print(listing)