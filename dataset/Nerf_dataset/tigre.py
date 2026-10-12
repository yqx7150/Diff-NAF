import torch
import pickle
import os
import sys
import numpy as np
from tigre.utilities.geometry import Geometry
import tigre

from torch.utils.data import DataLoader, Dataset
from pdb import set_trace as stx
import scipy.io
import random

class ConeGeometry_special(Geometry):




    def __init__(self, data):
        Geometry.__init__(self)



    def __init__(self, data):
        Geometry.__init__(self)



        self.DSD = data["DSD"] / 1000
        self.DSO = data["DSO"] / 1000

        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000
        self.sDetector = self.nDetector * self.dDetector

        self.nVoxel = np.array(data["nVoxel"][::-1])
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000
        self.sVoxel = self.nVoxel * self.dVoxel


        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000
        self.offDetector = np.array(
            [data["offDetector"][1], data["offDetector"][0], 0]) / 1000


        self.accuracy = data["accuracy"]

        self.mode = data["mode"]
        self.filter = data["filter"]


class ConeGeometry(object):



    def __init__(self, data):



        self.DSD = data["DSD"]/1000
        self.DSO = data["DSO"]/1000


        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"])/1000
        self.sDetector = self.nDetector * self.dDetector
        

        self.nVoxel = np.array(data["nVoxel"])
        self.dVoxel = np.array(data["dVoxel"])/1000
        self.sVoxel = self.nVoxel * self.dVoxel


        self.offOrigin = np.array(data["offOrigin"])/1000
        self.offDetector = np.array(data["offDetector"])/1000


        self.accuracy = data["accuracy"]

        self.mode = data["mode"]
        self.filter = data["filter"]



class TIGREDataset(Dataset):



    def __init__(self, path, n_rays=1024, type="train", device="cuda"):    
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)
        
        self.geo = ConeGeometry(data)
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)

        if type == "train":
            self.projs = torch.tensor(data["train"]["projections"], dtype=torch.float32, device=device)
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo, device)





            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1) 
            self.n_samples = data["numTrain"]

            coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                                                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device), indexing="ij"),
                                 -1)

            self.coords = torch.reshape(coords, [-1, 2])

            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)
        elif type == "val":
            self.projs = torch.tensor(data["val"]["projections"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1)
            self.n_samples = data["numVal"]
            self.image = torch.tensor(data["image"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":





            projs_valid = (self.projs[index]>0).flatten()   
            coords_valid = self.coords[projs_valid]
            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False)
            select_coords = coords_valid[select_inds].long()
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]
            projs = self.projs[index, select_coords[:, 0], select_coords[:, 1]]
            out = {
                "projs":projs,
                "rays":rays,
            }
        elif self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            out = {
                "projs":projs,
                "rays":rays,
            }
        return out

    def get_voxels(self, geo: ConeGeometry):














        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2









        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")



        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel
    






    def get_rays(self, angles, geo: ConeGeometry, device):
        




        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":






                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")







                





                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]






                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)






                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) 





                rays_o = pose[:3, -1].expand(rays_d.shape)

            elif geo.mode == "parallel":






                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  







                

                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]






                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)





                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1)









                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)

            
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)





    def angle2pose(self, DSO, angle):


        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])


        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        

        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)



        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T






    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):



















        




        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        


        dist_max = np.max([dist1, dist2, dist3, dist4])






        near = np.max([0, geo.DSO - dist_max - tolerance])





        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far












class TIGREDataset_update(Dataset):
    def __init__(self, cfg):    
        super().__init__()
        self.cfg = cfg
        matPath = cfg['exp']['datadir']
        self.generator(matPath)
        self.geo = ConeGeometry(cfg['data_generate']) 
        self.n_rays = cfg['train']['n_rays']
        self.near, self.far = self.get_near_far(self.geo)
        rays = self.get_rays(self.angles, self.geo,cfg['device'])     
        self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1) 
        coords = torch.stack(torch.meshgrid(torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=cfg['device']),
                                            torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=cfg['device']), indexing="ij"),-1)
        self.coords = torch.reshape(coords, [-1, 2])
        self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=cfg['device'])


        if self.cfg['is_require_eval_data'] is True:
            eval_rays = self.get_rays(self.eval_angles, self.geo,cfg['device'])     
            self.eval_rays =  torch.cat([eval_rays, torch.ones_like(eval_rays[...,:1])*self.near, torch.ones_like(eval_rays[...,:1])*self.far], dim=-1) 

        self.type = "train"

    def __len__(self):
        return self.projs_nums

    def __getitem__(self, index):
        if self.type == "val":
            rays = self.rays[index]
            projs = self.projs[index]
            label_projs = self.label_projs[index]
            out = {
                "projs":projs,
                "rays":rays,
                "label_projs":label_projs,
            }
        else:
            projs_valid = (self.projs[index]>0).flatten()   
            coords_valid = self.coords[projs_valid] 
            select_inds = np.random.choice(coords_valid.shape[0], size=[self.n_rays], replace=False) 
            select_coords = coords_valid[select_inds].long()        
            rays = self.rays[index, select_coords[:, 0], select_coords[:, 1]]   
            projs = self.projs[index, select_coords[:,0], select_coords[:,1]] 
            label_projs = self.label_projs[index, select_coords[:, 0], select_coords[:, 1]] 
            out = {
                "projs":projs,
                "rays":rays,
                "label_projs":label_projs,
            }
        return out

    def get_voxels(self, geo: ConeGeometry):

        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2


        xyz = np.meshgrid(np.linspace(-s1, s1, n1),
                        np.linspace(-s2, s2, n2),
                        np.linspace(-s3, s3, n3), indexing="ij")

        voxel = np.asarray(xyz).transpose([1, 2, 3, 0])
        return voxel

    def get_rays(self, angles, geo: ConeGeometry, device):


        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            rays_o, rays_d = None, None
            if geo.mode == "cone":
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                    torch.linspace(0, H - 1, H, device=device), indexing="ij")

                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)



                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1) 
                rays_o = pose[:3, -1].expand(rays_d.shape)

            elif geo.mode == "parallel":

                i, j = torch.meshgrid(torch.linspace(0, W - 1, W, device=device),
                                        torch.linspace(0, H - 1, H, device=device), indexing="ij")  

                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]

                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3,:3], dirs[..., None]).to(device), -1)
                rays_o = torch.sum(torch.matmul(pose[:3,:3], torch.stack([uu,vv,torch.zeros_like(uu)],-1)[..., None]).to(device), -1) + pose[:3, -1].expand(rays_d.shape)
            
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))

        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):

        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi1), -np.sin(phi1)],
                    [0.0, np.sin(phi1), np.cos(phi1)]])

        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                    [np.sin(phi2), np.cos(phi2), 0.0],
                    [0.0, 0.0, 1.0]])
        
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)

        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T


    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):

        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        

        dist_max = np.max([dist1, dist2, dist3, dist4])


        near = np.max([0, geo.DSO - dist_max - tolerance])

        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
    
    def convert_to_attenuation(self,data: np.array, rescale_slope: float, rescale_intercept: float):
        HU = data * rescale_slope + rescale_intercept
        mu_water = 0.206
        mu_air = 0.0004
        mu = mu_water + (mu_water - mu_air) / 1000 * HU
        return mu


    def loadImage(self,dirname, nVoxels, convert, rescale_slope, rescale_intercept, normalize=True):


        test_data = np.load(dirname) 
        image_ori = test_data.astype(np.float32)
        image_ori = (image_ori - image_ori.min()) / (image_ori.max() - image_ori.min())

        if convert:
            image =self.convert_to_attenuation(image_ori, rescale_slope, rescale_intercept)
        else:
            image = image_ori  

        imageDim = image.shape  











        image_max = np.max(image) 
        image_min = np.min(image) 
        image_mean = np.mean(image)  
        print("Range of CT image is [%f, %f], mean: %f" % (image_min, image_max, image_mean))

        if normalize and image_min !=0 and image_max != 1:
            print("Normalize range to [0, 1]")
            image = (image - image_min) / (image_max - image_min)  

        return image  


    def generator(self,npyPath):
        self.geo_sp = ConeGeometry_special(self.cfg['data_generate'])  
        img = self.loadImage(
            npyPath, 
            self.cfg['data_generate']["nVoxel"],
            self.cfg['data_generate']["convert"],
            self.cfg['data_generate']["rescale_slope"],
            self.cfg['data_generate']["rescale_intercept"],
            self.cfg['data_generate']["normalize"]
        ) 
        image = img.copy()
        self.ori_img = img
        maxsss = img.max()
        minsss = img.min()
        self.image = torch.tensor(image, dtype=torch.float32, device=self.cfg['device'])                 


        self.angles =  np.linspace(0, self.cfg['data_generate']["totalAngle"] / 180 * np.pi, self.cfg['data_generate']["numTrain"]+1)[:-1]
        self.init_angles =  np.linspace(0, self.cfg['data_generate']["totalAngle"] / 180 * np.pi, self.cfg['data_generate']["numTrain"]+1)[:-1]
        projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), self.geo_sp, self.angles)[:, ::-1, :].copy()

        

        maxsss = projections.max()
        minsss = projections.min()
        self.projs = torch.tensor(projections, dtype=torch.float32, device=self.cfg['device']) 
        self.label_projs = torch.tensor(projections, dtype=torch.float32, device=self.cfg['device']) 
        self.init_projs = torch.tensor(projections, dtype=torch.float32, device=self.cfg['device']) 
        self.projs_nums = len(self.projs)
        
        if self.cfg['is_require_eval_data'] is True:
            self.eval_angles =  np.linspace(0, self.cfg['data_generate']["totalAngle"] / 180 * np.pi, self.cfg['data_generate']["numVal"]+1)[:-1]
            eval_projections = tigre.Ax(np.transpose(img, (2, 1, 0)).copy(), self.geo_sp, self.eval_angles)[:, ::-1, :].copy()
            self.eval_projs = torch.tensor(eval_projections, dtype=torch.float32, device=self.cfg['device']) 

    def interpolate_angles(self , original_angles, factor=2):

        if factor < 2:
            raise ValueError("factor must be >= 2 (got {})".format(factor))
        
        angles_rad = original_angles
        
        deltas = angles_rad[1] - angles_rad[0]
        deltas = deltas / factor
        new_angles_rad = []
        for i in range(factor-1):
            new_angles_rad_temp = []
            new_angles_rad_temp = angles_rad + deltas*(i+1)
            new_angles_rad.extend(new_angles_rad_temp)

        new_angles_rad = sorted(new_angles_rad)
        
        return np.array(new_angles_rad)
    

    def update_val(self,multiple):

        total_angle_rad = np.radians(self.cfg['data_generate']["totalAngle"])
        start_angle_rad = np.radians(self.cfg['data_generate']["startAngle"])

        self.new_angles = self.interpolate_angles(self.init_angles, factor=multiple)





        projections = tigre.Ax(np.transpose(self.ori_img, (2, 1, 0)).copy(), self.geo_sp,self.new_angles)[:, ::-1, :].copy()

        self.label_projs = torch.tensor(projections, dtype=torch.float32, device=self.cfg['device']) 
        rays = self.get_rays(self.new_angles, self.geo,self.cfg['device'])     
        self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1) 
        self.n_samples = len(self.label_projs)
        
        self.type = "val"


    def update_train(self,new_proj):
        new_proj = torch.from_numpy(new_proj).to(self.projs.device) 

        combined_angles = np.concatenate([self.init_angles, self.new_angles])  
        combined_projs = torch.cat([self.init_projs, new_proj])          
        

        new_sorted_idx = np.argsort(combined_angles)  
        self.angles = combined_angles[new_sorted_idx]
        self.projs = combined_projs[new_sorted_idx]

        projections = tigre.Ax(np.transpose(self.ori_img, (2, 1, 0)).copy(), self.geo_sp, self.angles)[:, ::-1, :].copy()
        self.label_projs = torch.tensor(projections, dtype=torch.float32, device=self.cfg['device']) 
        rays = self.get_rays(self.angles, self.geo,self.cfg['device'])     
        self.rays = torch.cat([rays, torch.ones_like(rays[...,:1])*self.near, torch.ones_like(rays[...,:1])*self.far], dim=-1) 
        self.n_samples = len(self.projs)

        self.type = "train"
