import numpy as np
import h5py
from paddle.io import Dataset
import paddle as pd

def readsave(arg1,arg2):
    n1 = np.array(arg1[arg2][:])
    return n1

# define a random dataset
class TurbulenceDataset(Dataset):
    def __init__(self, h5_path, cutout, ratio: int = 4, num_samples: int = 10000, h5_size: int = 1024, scaler: float = 3):
        self.num_samples = num_samples
        self.h5_path = h5_path
        self.cutout = cutout
        self.ratio = ratio
        self.scaler = scaler
        # load the data from the h5 file
        self.data = h5py.File(self.h5_path, "r")
        self.frames_indices = [i for i in list(self.data.keys()) if i.startswith("Velocity")]
        self.num_frames = len(self.frames_indices)
        self.h5_size = h5_size

    
    def randomly_select_frame(self, idx):
        """
        randomly select a frame from the dataset
        """
        frame_idx = np.random.randint(0, self.num_frames)
        frame = self.data[self.frames_indices[frame_idx]]
        return frame[0,:,:,:]
    
    def randomly_select_cutout(self, frame):
        """
        randomly select a patch of the image from the frame
        """
        # randomly select a point in the frame
        x = np.random.randint(0, self.h5_size - self.cutout)
        y = np.random.randint(0, self.h5_size - self.cutout)
        # select the patch
        patch = frame[x:x+self.cutout, y:y+self.cutout, :]
        patch /= self.scaler
        # resize with average pool to get the low resolution image
        small_patch = np.mean(patch.reshape(self.cutout//self.ratio, self.ratio, self.cutout//self.ratio, self.ratio, 3), axis=(1, 3))
        # now convert to tensor and turn into CHW format
        patch = pd.to_tensor(patch).transpose([2, 0, 1])
        small_patch = pd.to_tensor(small_patch).transpose([2, 0, 1])
        return patch, small_patch

    def __getitem__(self, idx):
        frame = self.randomly_select_frame(idx)
        patch, small_patch = self.randomly_select_cutout(frame)
        return small_patch, patch

    def __len__(self):
        return self.num_samples

def get_energy(frame):
    """
    calculate the energy of the frame
    """
    frame = frame.numpy().transpose([1, 2, 0])
    energy = np.sum(np.square(frame), axis=2)
    return energy

def get_rgb(frame):
    frame = frame.numpy().transpose([1,2,0])
    min_val = np.min(frame, axis=2, keepdims=True)
    max_val = np.max(frame, axis=2, keepdims=True)
    frame = (frame - min_val) / (max_val - min_val)
    return frame


if __name__ == "__main__":
    # create a dataset
    dataset = TurbulenceDataset("data/data169495/isotropic1024coarse_10.h5", 128, 4, 10000)
    # create a matplotlib subplot grid of 4 * 2
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(4, 8))
    # iterate over the dataset
    for i in range(4):
        patch, small_patch = dataset[i]
        # plot the low resolution image
        axs[i, 0].imshow(get_energy(small_patch))
        # plot the high resolution image
        axs[i, 1].imshow(get_energy(patch))
    # show the plot
    plt.show()