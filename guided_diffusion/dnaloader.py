import torch
import torch.nn
import numpy as np
import os
import os.path
# import nibabel
import tifffile
from PIL import Image
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2

def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2.0*normalized_input - 1.0

# Initialize DNADataset and return data loader iter
def initialize_dataset(root_dir="", batch_size=2, src="/src/", target="/targ/", shuffle_required=True):
    ds = DNADataset(root_dir, src, target)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle_required
    )
    return loader


class DNADataset(torch.utils.data.Dataset):
    def __init__(self, directory, source_dr="/src/", target_dr="/targ/", conditional_dr="/cond/"):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        # print("raghu: directory=",directory,source_dr,target_dr)
        self.source_dir = directory + source_dr
        self.target_dir = directory + target_dr
        self.condtional = directory + conditional_dr
        self.directory = os.path.expanduser(self.source_dir)

        self.normalization = False
        self.ftype = "png"
        self.transform_req = False
        
        self.seqtypes = ['bf', 'flo', 'cond']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        self.transform = A.Compose([
            A.RandomCrop(width=128, height=128),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomRotate90(),
            ToTensorV2()
        ],
        additional_targets={'image0': 'image', 'image1': 'image'})
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                # extract all files as channels
                for f in files:
                    datapoint = dict()
                    for i in self.seqtypes:
                        if i == 'bf':
                            datapoint[i] = os.path.join(self.source_dir, f)
                            # print("bf=",os.path.join(directory, self.source_dir, f))
                        if i == 'flo':
                            datapoint[i] = os.path.join(self.target_dir, f)
                            # print("flo=",os.path.join(directory, self.target_dir, f))
                        if i == 'cond':
                            datapoint[i] = os.path.join(self.condtional, f)
                    assert set(datapoint.keys()) == self.seqtypes_set, \
                        f'datapoint is incomplete, keys are {datapoint.keys()}'
                    self.database.append(datapoint)
                    # print(len(self.database))
        print(len(self.database))

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        if self.transform_req == False:
          for seqtype in self.seqtypes:
              if self.ftype == "tiff":
                  np_frame = tifffile.imread(filedict[seqtype])
                  if self.normalization:
                      np_frame = normalize_negative_one(np_frame)
                  path=filedict[seqtype]
                  out.append(torch.tensor(np_frame))
              else:
                im_frame = Image.open(filedict[seqtype])
                np_frame = np.array(im_frame)
                np_frame = (np_frame.astype(np.float32)/127.5) - 1
                path=filedict[seqtype]
                out.append(torch.tensor(np_frame))
          out = torch.stack(out)
          bf = out[:-2, ...]
          flo = out[-2, ...]
          cond = out[-1, ...][None, ...]
          flo = torch.where(flo > 0, flo, flo).float()
          cond = torch.where(cond > 0, cond, cond).float()
          return (bf, flo, cond)
        else:
          sf = np.array(Image.open(filedict[self.seqtypes[0]]))
          tf = np.array(Image.open(filedict[self.seqtypes[1]]))
          conditioned = np.array(Image.open(filedict[self.seqtypes[2]]))
          transformed = self.transform(image=sf,image0=tf, image1=conditioned)
          bf = transformed['image']/127.5 - 1.0
          flo = transformed['image0']/127.5 - 1.0
          cond = transformed['image1']/127.6 - 1.0
          return (bf, flo, cond)

    def __len__(self):
        return len(self.database)

