from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
import time
import cupy
import cupy as cp

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.CUDA = torch.cuda.is_available()
        self.TORCH = True
        # self.CUDA = False
    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tict = time.perf_counter()
        i_mesh = self.data_list.iloc[idx][0] #vtk file name
        
        # read vtk
        mesh = load(i_mesh)

        labels = mesh.celldata['Label'].astype('int32').reshape(-1, 1)

        # move mesh to origin
        N = mesh.NCells()
        # cells = np.zeros([N, 9], dtype='float32')

        tic = time.perf_counter()
        points = vtk2numpy(mesh.polydata().GetPoints().GetData())
        ids = vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((N, -1))[:,1:]
        cells = points[ids].reshape(N, 9)
        toc = time.perf_counter()
        # print(f"mesh dataset points 1 : {toc - tic:0.5f} seconds")

        tic = time.perf_counter()
        mean_cell_centers = mesh.centerOfMass()
        if not self.CUDA:
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]
        else:
            #cupy
            mean_cell_centers = cp.asarray(mean_cell_centers)
            cells = cupy.asarray(cells)
            cells[:, 0:3] -= mean_cell_centers[0:3]
            cells[:, 3:6] -= mean_cell_centers[0:3]
            cells[:, 6:9] -= mean_cell_centers[0:3]
        toc = time.perf_counter()
        # print(f"mesh dataset points 2 : {toc - tic:0.5f} seconds")
        tic = time.perf_counter()
        if not self.CUDA:
            # customized normal calculation; the vtk/vedo build-in function will change number of points
            v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = np.cross(v1, v2)
            mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            mesh.celldata['Normal'] = mesh_normals
        else:
        #cupy
            v1 = cupy.zeros([mesh.NCells(), 3], dtype='float32')
            v2 = cupy.zeros([mesh.NCells(), 3], dtype='float32')
            v1[:, 0] = cells[:, 0] - cells[:, 3]
            v1[:, 1] = cells[:, 1] - cells[:, 4]
            v1[:, 2] = cells[:, 2] - cells[:, 5]
            v2[:, 0] = cells[:, 3] - cells[:, 6]
            v2[:, 1] = cells[:, 4] - cells[:, 7]
            v2[:, 2] = cells[:, 5] - cells[:, 8]
            mesh_normals = cupy.cross(v1, v2)
            mesh_normal_length = cupy.linalg.norm(mesh_normals, axis=1)
            mesh_normals[:, 0] /= mesh_normal_length[:]
            mesh_normals[:, 1] /= mesh_normal_length[:]
            mesh_normals[:, 2] /= mesh_normal_length[:]
            mesh_normals_numpy =  cupy.asnumpy(mesh_normals)
            mesh.celldata['Normal'] = mesh_normals_numpy
        toc = time.perf_counter()
        # print(f"mesh dataset points 3 : {toc - tic:0.5f} seconds")

        tic = time.perf_counter()
        if not self.CUDA:
            # preprae input and make copies of original data
            points = mesh.points().copy()
            points[:, 0:3] -= mean_cell_centers[0:3]
            normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
            barycenters = mesh.cellCenters() # don't need to copy
            barycenters -= mean_cell_centers[0:3]
            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)
        else:
            #cupy
            points = mesh.points().copy()
            points = cupy.asarray(points)
            points[:, 0:3] -= mean_cell_centers[0:3]
            normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
            normals = cupy.asarray(normals)
            barycenters = mesh.cellCenters() # don't need to copy
            barycenters = cupy.asarray(barycenters)
            barycenters -= mean_cell_centers[0:3]
            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)
        toc = time.perf_counter()
        # print(f"mesh dataset points 4 : {toc - tic:0.5f} seconds")
        tic = time.perf_counter()
        if not self.CUDA:
            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))
            Y = labels

            # initialize batch of input and label
            X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
            Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
            S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
            S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

            # calculate number of valid cells (tooth instead of gingiva)
            positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
            negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

            num_positive = len(positive_idx) # number of selected tooth cells

            if num_positive > self.patch_size: # all positive_idx in this patch
                num_negative = int(self.patch_size * 0.3)
                left_positive = self.patch_size - num_negative
                positive_selected_idx = np.random.choice(positive_idx, size=left_positive, replace=False)
                negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))
            else:   # patch contains all positive_idx and some negative_idx
                num_negative = self.patch_size - num_positive # number of selected gingiva cells
                positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
                negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

            selected_idx = np.sort(selected_idx, axis=None)

            X_train[:] = X[selected_idx, :]
            Y_train[:] = Y[selected_idx, :]
        else:
            #cupy
            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = cupy.column_stack((cells, barycenters, normals))
            Y = cupy.asarray(labels)

            # initialize batch of input and label
            X_train = cupy.zeros([self.patch_size, X.shape[1]], dtype='float32')
            Y_train = cupy.zeros([self.patch_size, Y.shape[1]], dtype='int32')
            S1 = cupy.zeros([self.patch_size, self.patch_size], dtype='float32')
            S2 = cupy.zeros([self.patch_size, self.patch_size], dtype='float32')

            # calculate number of valid cells (tooth instead of gingiva)
            labels_cupy = cupy.asarray(labels)
            positive_idx = cupy.argwhere(labels_cupy>0)[:, 0] #tooth idx
            negative_idx = cupy.argwhere(labels_cupy==0)[:, 0] # gingiva idx

            num_positive = len(positive_idx) # number of selected tooth cells

            if num_positive > self.patch_size: # all positive_idx in this patch
                num_negative = int(self.patch_size * 0.3)
                left_positive = self.patch_size - num_negative
                positive_selected_idx = cupy.random.choice(positive_idx, size=left_positive, replace=False)
                negative_selected_idx = cupy.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = cupy.concatenate((positive_selected_idx, negative_selected_idx))
            else:   # patch contains all positive_idx and some negative_idx
                num_negative = self.patch_size - num_positive # number of selected gingiva cells
                positive_selected_idx = cupy.random.choice(positive_idx, size=num_positive, replace=False)
                negative_selected_idx = cupy.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = cupy.concatenate((positive_selected_idx, negative_selected_idx))

            selected_idx = cupy.sort(selected_idx, axis=None)

            X_train[:] = X[selected_idx, :]
            Y_train[:] = Y[selected_idx, :]
        toc = time.perf_counter()
        # print(f"mesh dataset points 5 : {toc - tic:0.5f} seconds")
        tic6 = time.perf_counter()
        if not self.CUDA:
            ticd = time.perf_counter()
            D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
            tocd = time.perf_counter()
            print(f"distance_matrix : {tocd - ticd:0.5f} seconds")
            S1[D<0.1] = 1.0
            S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

            S2[D<0.2] = 1.0
            S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

            X_train = X_train.transpose(1, 0)
            Y_train = Y_train.transpose(1, 0)

            sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                    'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}
        else:
            if not self.TORCH:
                ticd = time.perf_counter()
                D = distance_matrix(cupy.asnumpy(X_train[:, 9:12]), cupy.asnumpy(X_train[:, 9:12]))
                tocd = time.perf_counter()
                print(f"cpu distance_matrix : {tocd - ticd:0.5f} seconds")
            else:
                ticd = time.perf_counter()
                tX = torch.as_tensor(X_train[:, 9:12], device='cuda')
                D = torch.cdist(tX, tX)
                tocd = time.perf_counter()
                # print(f"torch distance_matrix : {tocd - ticd:0.5f} seconds")


            D = cupy.asarray(D)
            S1[D<0.1] = 1.0
            S1 = S1 / cupy.dot(cupy.sum(S1, axis=1, keepdims=True), cupy.ones((1, self.patch_size)))

            S2[D<0.2] = 1.0
            S2 = S2 / cupy.dot(cupy.sum(S2, axis=1, keepdims=True), cupy.ones((1, self.patch_size)))

            X_train = X_train.transpose(1, 0)
            Y_train = Y_train.transpose(1, 0)

            X_train_cpu = cupy.asnumpy(X_train)
            Y_train_cpu = cupy.asnumpy(Y_train)
            S1_cpu = cupy.asnumpy(S1)
            S2_cpu = cupy.asnumpy(S2)


            sample = {'cells': torch.from_numpy(X_train_cpu), 'labels': torch.from_numpy(Y_train_cpu),
                    'A_S': torch.from_numpy(S1_cpu), 'A_L': torch.from_numpy(S2_cpu)}
        toc6 = time.perf_counter()
        # print(f"mesh dataset points 6 : {toc6 - tic6:0.5f} seconds")

        toct =  time.perf_counter()
        # print(f"load and  preprocessing : {toct - tict:0.5f} seconds")

        return sample


if __name__ == '__main__':
    import os
    num_classes = 17
    data_folder = '/proj/MeshSegNet'
    train_list = os.path.join(data_folder, 'train_list_1.csv') # use 1-fold as example
    training_dataset = Mesh_Dataset(data_list_path=train_list, num_classes=num_classes, patch_size=6000)
    for i in range(10):
        y = training_dataset[i]