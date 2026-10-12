import argparse
import sys


if __name__ == "__main__":
    _stage_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _stage_parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    _stage_args, _stage_remaining_args = _stage_parser.parse_known_args()
    _selected_stage = _stage_args.stage
    sys.argv = [sys.argv[0], *_stage_remaining_args]

    if _selected_stage == 1:
        import argparse  
        import os


        def config_parser():
            parser = argparse.ArgumentParser()
            parser.add_argument("--gpu_id", default="0")
            parser.add_argument("--config", default=f"./config/Nerf_config/Lineformer/L067/test.yaml")
            parser.add_argument("--phase", default=f"Nerf_Train")
            return parser

        parser = config_parser()
        args = parser.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        import datetime
        from tqdm import tqdm
        import numpy as np
        import torch
        from model.Nerf.Nerf_loss import calc_mse_loss
        from shutil import copyfile
        import imageio.v2 as iio
        import yaml

        from model.Nerf.Nerf_network import get_network
        from model.Nerf.Nerf_encoder import get_encoder
        from dataset.Nerf_dataset import TIGREDataset_update 
        from utilis.Nerf.Nerf_render import render, run_network
        from utilis.Nerf.Nerf_utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image, get_mse,gen_log, time2file_name



        from configloading import load_config



        class Trainer:
            def __init__(self, cfg, device="cuda"):



                self.iter_num = 1
                date_time = str(datetime.datetime.now())
                self.date_time = time2file_name(date_time)
                self.rootpath = os.path.join('logs',cfg['phase'], cfg['exp']['expdir'], self.date_time)
                self.expdir = os.path.join(self.rootpath,'Nerf_model_save')
                self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
                self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
                self.ckpt_best_dir = os.path.join(self.expdir, "ckpt_best.tar")
                self.evaldir = os.path.join(self.rootpath, "Nerf_eval")
                self.diffiusiondir = os.path.join(self.rootpath, "diffiusion")
                os.makedirs(self.expdir, exist_ok=True)
                os.makedirs(self.evaldir, exist_ok=True)
                os.makedirs(self.diffiusiondir, exist_ok=True)
                self.logger = gen_log(self.rootpath)




                self.conf = cfg       
                network = get_network(cfg["network"]["net_type"])
                cfg["network"].pop("net_type", None)
                encoder = get_encoder(**cfg["encoder"])
                self.net = network(encoder, **cfg["network"]).to(device)
                grad_vars = list(self.net.parameters())
                self.net_fine = None
                if self.conf["render"]["n_fine"] > 0:
                    self.net_fine = network(encoder, **cfg["network"]).to(device)
                    grad_vars += list(self.net_fine.parameters())
                self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])
                self.best_psnr_3d = 0

                if cfg['phase'] == 'Nerf_Test':
                    weights_path = os.path.join(cfg['test']['pre_model'])
                    ckpt = torch.load(weights_path,map_location=device)
                    self.net.load_state_dict(ckpt["network"])
                    self.net.eval()
                    if self.conf["render"]["n_fine"] > 0:
                        self.net_fine.load_state_dict(ckpt["network_fine"])
                        self.net_fine.eval()






                self.conf['is_require_eval_data'] = True
                self.train_dset = TIGREDataset_update(self.conf) 
                self.train_dloader = torch.utils.data.DataLoader(self.train_dset, batch_size=self.conf["train"]["n_batch"]) 
                self.voxels = self.train_dset.voxels 




                formatted_cfg = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                self.logger.info("\n" + formatted_cfg) 



            def Nerf_train(self):



                for idx_epoch in range(self.conf["train"]["epoch"]+1):
                    if (idx_epoch % self.conf["log"]["i_eval"] == 0 or idx_epoch == self.conf["train"]["epoch"]) :
                        self.net.eval()             
                        with torch.no_grad():
                            loss_test = self.eval_step(idx_epoch=idx_epoch)
                        self.net.train()
                        self.logger.info("----Nerf_EVAL---- epoch: {}/{}{}".format(idx_epoch, self.conf["train"]["epoch"], self.fmt_loss_str(loss_test)))



                    for data in self.train_dloader:
                        self.net.train()
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(data)
                        loss.backward()
                        self.optimizer.step()
                        loss_train = loss.item()



                    if idx_epoch % 50 == 0:
                        self.logger.info("epoch={}/{}, loss={:.4g}, lr={:.4g}".format(idx_epoch,self.conf["train"]["epoch"],loss_train,self.optimizer.param_groups[0]['lr']))
                    if (idx_epoch % self.conf["log"]["i_save"]== 0 or idx_epoch == self.conf["train"]["epoch"]) and self.conf["log"]["i_save"] > 0 and idx_epoch > 0:
                        if os.path.exists(self.ckptdir):
                            copyfile(self.ckptdir, self.ckptdir_backup)
                        self.logger.info("----Nerf_SAVE---- epoch: {}/{}, path: {}".format(idx_epoch,self.conf["train"]["epoch"],self.ckptdir))
                        torch.save(
                            {
                                "epoch": idx_epoch,
                                "network": self.net.state_dict(),
                                "network_fine": self.net_fine.state_dict() if self.conf["render"]["n_fine"] > 0 else None,
                                "optimizer": self.optimizer.state_dict(),
                            },
                            self.ckptdir,
                        ) 
                    self.lr_scheduler.step()


            def Nerf_output(self):
                with torch.no_grad():
                    projs = self.train_dset.eval_projs
                    rays = self.train_dset.eval_rays.reshape(-1, 8)
                    N, H, W = projs.shape
                    projs_pred = []

                    n_rays = cfg["train"]["n_rays"]
                    netchunk = cfg["render"]["netchunk"]
                    self.logger.info("Nerf 生成新投影")

                    for i in tqdm(range(0, rays.shape[0], n_rays)):    
                        projs_pred.append(render(rays[i:i+n_rays], self.net, self.net_fine, **cfg["render"])["acc"])

                    projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W) 
                    image = self.train_dset.image
                    image_pred = run_network(self.train_dset.voxels, self.net if self.net_fine is not None else self.net, netchunk)
                    image_pred = image_pred.squeeze()

                    loss = {
                        "proj_psnr": get_psnr(projs_pred, projs),

                        "psnr_3d": get_psnr_3d(image_pred, image),

                    }
                    self.logger.info(loss)

                    resdir = os.path.join(self.evaldir,'the_last_eval')  
                    proj_pred_dir = os.path.join(resdir, "proj_pred")
                    proj_gt_dir = os.path.join(resdir, "proj_gt")

                    ct_pred_dir_H = os.path.join(resdir, "CT", "H", "ct_pred")
                    ct_gt_dir_H = os.path.join(resdir, "CT", "H", "ct_gt")
                    ct_pred_dir_W = os.path.join(resdir, "CT", "W", "ct_pred")
                    ct_gt_dir_W = os.path.join(resdir, "CT", "W", "ct_gt")
                    ct_pred_dir_L = os.path.join(resdir, "CT", "L", "ct_pred")
                    ct_gt_dir_L = os.path.join(resdir, "CT", "L", "ct_gt")

                    H, W, L = image_pred.shape
                    self.logger.info(image_pred.shape)

                    os.makedirs(proj_pred_dir, exist_ok=True)
                    os.makedirs(proj_gt_dir, exist_ok=True)
                    os.makedirs(ct_pred_dir_H, exist_ok=True)
                    os.makedirs(ct_gt_dir_H, exist_ok=True)
                    os.makedirs(ct_pred_dir_W, exist_ok=True)
                    os.makedirs(ct_gt_dir_W, exist_ok=True)
                    os.makedirs(ct_pred_dir_L, exist_ok=True)
                    os.makedirs(ct_gt_dir_L, exist_ok=True)

                    for i in tqdm(range(N)):
                        iio.imwrite(os.path.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((cast_to_image(projs_pred[i]))*255).astype(np.uint8))
                        iio.imwrite(os.path.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((cast_to_image(projs[i]))*255).astype(np.uint8))
            
                    for i in tqdm(range(H)):
                        iio.imwrite(os.path.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

                    for i in tqdm(range(W)):
                        iio.imwrite(os.path.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

                    for i in tqdm(range(L)):
                        iio.imwrite(os.path.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))

                npy = os.path.join(resdir, 'npy')
                os.makedirs(npy, exist_ok=True)
                np.save(os.path.join(npy, 'pred.npy'), projs_pred.cpu().numpy())
                np.save(os.path.join(npy, 'label.npy'), projs.cpu().numpy())

                np.save(os.path.join(npy, 'image_pred.npy'), image_pred.cpu().numpy())
                np.save(os.path.join(npy, 'image.npy'), image.cpu().numpy())
                return projs_pred.cpu().numpy(),projs.cpu().numpy()

            def eval_step(self, idx_epoch):
                projs = self.train_dset.eval_projs
                rays = self.train_dset.eval_rays  
                rays = rays.reshape(-1, 8)
                N, H, W = projs.shape
                projs_pred = []


                for i in tqdm(range(0, rays.shape[0], self.conf["train"]["n_rays"])):
                    chunk_rays = rays[i:i + self.conf["train"]["n_rays"]]
                    ret = render(chunk_rays, self.net, self.net_fine,  ** self.conf["render"])
                    projs_pred.append(ret["acc"])


                projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W)
                image = self.train_dset.image
                image_pred = run_network(self.train_dset.voxels,
                                         self.net_fine if self.net_fine else self.net,
                                         self.conf["render"]["netchunk"])
                image_pred = image_pred.squeeze()  

                loss = {
                        "proj_psnr": get_psnr(projs_pred, projs),

                        "psnr_3d": get_psnr_3d(image_pred, image),

                    }





                if loss["psnr_3d"] > self.best_psnr_3d:
                    torch.save({
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.conf["render"]["n_fine"] else None,
                        "optimizer": self.optimizer.state_dict(),
                    }, self.ckpt_best_dir)
                    self.best_psnr_3d = loss["psnr_3d"]
                    self.logger.info(f"最佳模型更新，轮次:{idx_epoch}, 最佳3D PSNR:{self.best_psnr_3d:.4g}")

                resdir = os.path.join(self.evaldir, f"epoch_{idx_epoch:05d}")
                os.makedirs(resdir, exist_ok=True)
                proj_pred_dir = os.path.join(resdir, "proj_pred")
                proj_gt_dir = os.path.join(resdir, "proj_gt")

                ct_pred_dir_H = os.path.join(resdir, "CT", "H", "ct_pred")
                ct_gt_dir_H = os.path.join(resdir, "CT", "H", "ct_gt")
                ct_pred_dir_W = os.path.join(resdir, "CT", "W", "ct_pred")
                ct_gt_dir_W = os.path.join(resdir, "CT", "W", "ct_gt")
                ct_pred_dir_L = os.path.join(resdir, "CT", "L", "ct_pred")
                ct_gt_dir_L = os.path.join(resdir, "CT", "L", "ct_gt")

                H, W, L = image_pred.shape
                self.logger.info(image_pred.shape)

                os.makedirs(proj_pred_dir, exist_ok=True)
                os.makedirs(proj_gt_dir, exist_ok=True)
                os.makedirs(ct_pred_dir_H, exist_ok=True)
                os.makedirs(ct_gt_dir_H, exist_ok=True)
                os.makedirs(ct_pred_dir_W, exist_ok=True)
                os.makedirs(ct_gt_dir_W, exist_ok=True)
                os.makedirs(ct_pred_dir_L, exist_ok=True)
                os.makedirs(ct_gt_dir_L, exist_ok=True)

                for i in tqdm(range(N)):
                    iio.imwrite(os.path.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((cast_to_image(projs_pred[i]))*255).astype(np.uint8))
                    iio.imwrite(os.path.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((cast_to_image(projs[i]))*255).astype(np.uint8))
        
                for i in tqdm(range(H)):
                    iio.imwrite(os.path.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

                for i in tqdm(range(W)):
                    iio.imwrite(os.path.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

                for i in tqdm(range(L)):
                    iio.imwrite(os.path.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))

                npy = os.path.join(resdir, 'npy')
                os.makedirs(npy, exist_ok=True)
                np.save(os.path.join(npy, 'pred.npy'), projs_pred.cpu().numpy())
                np.save(os.path.join(npy, 'label.npy'), projs.cpu().numpy())

                np.save(os.path.join(npy, 'image_pred.npy'), image_pred.cpu().numpy())
                np.save(os.path.join(npy, 'image.npy'), image.cpu().numpy())

                return loss
    

            def fmt_loss_str(self,losses):
                return "".join(", " + k + ": " + f"{losses[k].item():.4g}" for k in losses)


            def compute_loss(self, data):
                rays = data["rays"].reshape(-1, 8)  
                projs = data["projs"].reshape(-1)  
                ret = render(rays, self.net, self.net_fine,  ** self.conf["render"]) 
                projs_pred = ret["acc"] 
                maxp =  projs_pred.max()
                minp =  projs_pred.min()
                loss = {"loss": 0.}
                calc_mse_loss(loss, projs, projs_pred)
                return loss["loss"]
    

        if __name__ == "__main__":
            cfg = load_config(args.config)
            device = torch.device(f"cuda:0")
            cfg['device'] = device
            cfg['phase'] = args.phase
            trainer = Trainer(cfg)
            if cfg['phase'] == 'Nerf_Train' :
                trainer.Nerf_train()  
            elif cfg['phase'] == 'Nerf_Test' :
                trainer.Nerf_output()  

    elif _selected_stage == 2:
        import argparse  
        import os


        def config_parser():
            parser = argparse.ArgumentParser()

            parser.add_argument("--config", default=f"./config/diffiusion/test.yaml")

            parser.add_argument("--gpu_id", default="0", help="使用的GPU编号")
            parser.add_argument("--phase", default=f"diffiusion_Train")
            return parser


        parser = config_parser()
        args = parser.parse_args()


        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id



        from model.diffiusion.residual_denoising_diffusion_pytorch import ResidualDiffusion,Trainer, UnetRes
        import torch
        from pathlib import Path
        import datetime
        from utilis.Nerf.Nerf_utils  import gen_log, time2file_name
        from configloading import load_config
        import yaml



        cfg = load_config(args.config)

        cfg['device'] = torch.device("cuda:0")


        if __name__ == "__main__":

            cfg['phase'] = args.phase
            date_time = str(datetime.datetime.now())
            date_time = time2file_name(date_time) 
            save_path = os.path.join("logs", cfg['phase'],date_time)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            cfg['save_path'] = save_path
            logger = gen_log(save_path)
            formatted_cfg = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
            logger.info("\n" + formatted_cfg) 



            model = UnetRes(cfg)
            diffusion = ResidualDiffusion(model,cfg)
            trainer = Trainer(diffusion, cfg,logger,save_path)


            if cfg['phase'] == 'diffiusion_Train' :
                trainer.train()
            elif cfg['phase'] == 'diffiusion_Test' :
                trainer.load()
                trainer.test(last=True)

    elif _selected_stage == 3:
        import argparse  
        import os


        def config_parser():
            parser = argparse.ArgumentParser()
            parser.add_argument("--gpu_id", default="0")
            parser.add_argument("--config1", default=f"./config/Nerf_config/Lineformer/L067/test.yaml")
            parser.add_argument("--config2", default=f"./config/diffiusion/test.yaml")
            parser.add_argument("--cycle", default = 3)
            parser.add_argument("--phase", default=f"Iteration")
            parser.add_argument("--is_lianguanxing", default=False)


            return parser

        parser = config_parser()
        args = parser.parse_args()

        os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        import datetime
        import argparse
        from tqdm import tqdm
        import numpy as np
        import torch
        from model.Nerf.Nerf_loss import calc_mse_loss
        from shutil import copyfile
        from model.diffiusion.residual_denoising_diffusion_pytorch import ResidualDiffusion,Trainer, UnetRes
        import imageio.v2 as iio
        import yaml

        from model.Nerf.Nerf_network import get_network
        from model.Nerf.Nerf_encoder import get_encoder
        from dataset.Nerf_dataset import TIGREDataset_update 
        from utilis.Nerf.Nerf_render import render, run_network
        from utilis.Nerf.Nerf_utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image, get_mse,gen_log, time2file_name,get_psnr_slice



        from configloading import load_config
        import torch
        import numpy as np

        def normalize_samples_global(data,cfg):
            if cfg['diffusion_data']['norm_type'] == 'globe_norm':
                pre, gt = data[0], data[1]
                global_min = np.min(pre)
                global_max = np.max(pre)
                global_range = global_max - global_min + 1e-12  
                normalized_pre = (pre - global_min) / global_range
                normalized_gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
                params = {
                    'min_val': global_min,
                    'max_val': global_max,
                    'range_val': global_range
                }
                return (normalized_pre, normalized_gt), params
    

            elif cfg['diffusion_data']['norm_type'] == 'single_norm':
                pre,gt = data[0],data[1]
                params = np.zeros((len(pre), 2))  
                normalized_data = np.zeros_like(pre)
                normalized_gt = np.zeros_like(gt)
                for i in range(len(pre)):
                    sample = pre[i]  
                    gt_sample = gt[i]
                    min_val = np.min(sample)
                    max_val = np.max(sample)
                    range_val = max_val - min_val + 1e-12  
                    normalized_data[i] = (sample - min_val) / range_val
                    normalized_gt[i] = (gt_sample - np.min(gt_sample)) / (np.max(gt_sample) - np.min(gt_sample) + 1e-12)
                    params[i] = [min_val, max_val]
                return (normalized_data,normalized_gt), params
    
            elif cfg['diffusion_data']['norm_type'] == 'Increase_tenfold':
                pre,gt = data[0],data[1]
                pre = pre * cfg['diffusion_data']['scale_factor']
                gt =gt * cfg['diffusion_data']['scale_factor']
                params = 0
                return (pre,gt), params
            else:
                pre,gt = data[0],data[1]
                params = 0
                return (pre,gt), params

        def denormalize_samples_global(normalized_data, params,cfg):
            if cfg['diffusion_data']['norm_type'] == 'globe_norm':
                min_val = params['min_val']
                range_val = params['range_val']
        
                denormalized_data = normalized_data * range_val + min_val
                return denormalized_data
            elif cfg['diffusion_data']['norm_type'] == 'single_norm':
                denormalized_data = np.zeros_like(normalized_data)
                for i in range(len(normalized_data)):
                    min_val, max_val = params[i]
                    range_val = max_val - min_val
            

                    denormalized_data[i] = normalized_data[i] * range_val + min_val
        
                return denormalized_data
            elif cfg['diffusion_data']['norm_type'] == 'Increase_tenfold':
                return normalized_data /cfg['diffusion_data']['scale_factor']
            else:
                return normalized_data



        class iterator:
            def __init__(self, cfg, device="cuda"):



                self.iter_num = 1
                cfg['is_require_eval_data'] = False
                date_time = str(datetime.datetime.now())
                self.date_time = time2file_name(date_time)
                self.rootpath = os.path.join('logs',cfg['phase'], cfg['exp']['expdir'],  self.date_time,'iter_num_{}'.format(self.iter_num))
                self.expdir = os.path.join(self.rootpath,'Nerf_model_save')
                self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
                self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
                self.ckpt_best_dir = os.path.join(self.expdir, "ckpt_best.tar")
                self.evaldir = os.path.join(self.rootpath, "Nerf_eval")
                self.diffiusiondir = os.path.join(self.rootpath, "diffiusion")
                os.makedirs(self.expdir, exist_ok=True)
                os.makedirs(self.evaldir, exist_ok=True)
                os.makedirs(self.diffiusiondir, exist_ok=True)
                self.logger = gen_log(self.rootpath)

        
        



                self.conf = cfg       
                network = get_network(cfg["network"]["net_type"])
                cfg["network"].pop("net_type", None)
                encoder = get_encoder(**cfg["encoder"])
                self.net = network(encoder, **cfg["network"]).to(device)
                grad_vars = list(self.net.parameters())
                self.net_fine = None
                if self.conf["render"]["n_fine"] > 0:
                    self.net_fine = network(encoder, **cfg["network"]).to(device)
                    grad_vars += list(self.net_fine.parameters())
                self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])
                self.best_psnr_3d = 0





        
                diffiusion_model = UnetRes(cfg)
                diffiusion_diffusion = ResidualDiffusion(diffiusion_model,cfg)
                self.diffiusion_model = Trainer(diffiusion_diffusion, cfg,self.logger,self.diffiusiondir)

                if args.is_lianguanxing :
                    self.diffiusiondir_64 = os.path.join(self.rootpath, "diffiusion_64")
                    os.makedirs(self.diffiusiondir_64, exist_ok=True)
                    cfg_64diffiusion = load_config('./config/diffiusion/AAPMDR_64_line.yaml')
                    cfg_64diffiusion['device'] = cfg['device']
                    cfg_64diffiusion['phase'] = cfg['phase'] 
                    diffiusion_model_64 = UnetRes(cfg_64diffiusion)
                    diffiusion_diffusion_64 = ResidualDiffusion(diffiusion_model_64,cfg_64diffiusion)
                    self.diffiusion_model_64 = Trainer(diffiusion_diffusion_64, cfg_64diffiusion,self.logger,self.diffiusiondir_64)





                self.train_dset = TIGREDataset_update(cfg) 
                self.train_dloader = torch.utils.data.DataLoader(self.train_dset, batch_size=cfg["train"]["n_batch"]) 
                self.voxels = self.train_dset.voxels
    
                formatted_cfg = yaml.dump(cfg, default_flow_style=False, sort_keys=False)
                self.logger.info("\n" + formatted_cfg)
            def Nerf_train(self):



                for idx_epoch in range(self.conf["train"]["epoch"]+1):
                    if (idx_epoch % self.conf["log"]["i_eval"] == 0 or idx_epoch == self.conf["train"]["epoch"]) :
                        self.net.eval()             
                        with torch.no_grad():
                            loss_test = self.eval_step(idx_epoch=idx_epoch)
                        self.net.train()
                        self.logger.info("----Nerf_EVAL---- epoch: {}/{}{}".format(idx_epoch, self.conf["train"]["epoch"], self.fmt_loss_str(loss_test)))



                    for data in self.train_dloader:
                        self.net.train()
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(data)
                        loss.backward()
                        self.optimizer.step()
                        loss_train = loss.item()



                    if idx_epoch % 50 == 0:
                        self.logger.info("epoch={}/{}, loss={:.4g}, lr={:.4g}".format(idx_epoch,self.conf["train"]["epoch"],loss_train,self.optimizer.param_groups[0]['lr']))
                    if (idx_epoch % self.conf["log"]["i_save"]== 0 or idx_epoch == self.conf["train"]["epoch"]) and self.conf["log"]["i_save"] > 0 and idx_epoch > 0:
                        if os.path.exists(self.ckptdir):
                            copyfile(self.ckptdir, self.ckptdir_backup)
                        self.logger.info("----Nerf_SAVE---- epoch: {}/{}, path: {}".format(idx_epoch,self.conf["train"]["epoch"],self.ckptdir))
                        torch.save(
                            {
                                "epoch": idx_epoch,
                                "network": self.net.state_dict(),
                                "network_fine": self.net_fine.state_dict() if self.conf["render"]["n_fine"] > 0 else None,
                                "optimizer": self.optimizer.state_dict(),
                            },
                            self.ckptdir,
                        ) 
                    self.lr_scheduler.step()


            def Nerf_output(self,multiple):
                self.train_dset.update_val(multiple)
                self.logger.info("--------------------------------------------------")
                self.logger.info("-----更新新角度验证集------")
                self.logger.info(f"-----验证集角度数------  {len(self.train_dset.new_angles)}")
                self.logger.info(f"{self.train_dset.new_angles}")
                self.logger.info("--------------------------------------------------")

                if multiple>2:
                    ckpt = torch.load(self.ckpt_best_dir,map_location=device)
                    self.net.load_state_dict(ckpt["network"])
                    self.net.eval()
                    if self.conf["render"]["n_fine"] > 0:
                        self.net_fine.eval()
                with torch.no_grad():
                    projs = self.train_dset.label_projs  
                    rays = self.train_dset.rays.reshape(-1, 8)  
                    N, H, W = projs.shape
                    projs_pred = []
                    n_rays = cfg["train"]["n_rays"]
                    netchunk = cfg["render"]["netchunk"]
                    self.logger.info("Nerf 生成新投影")

                    for i in tqdm(range(0, rays.shape[0], n_rays)):    
                        projs_pred.append(render(rays[i:i+n_rays], self.net, self.net_fine, **cfg["render"])["acc"])

                    projs_pred = torch.cat(projs_pred, 0).reshape(N, H, W) 


                    image = self.train_dset.image
                    image_pred = run_network(self.train_dset.voxels, self.net if self.net_fine is not None else self.net, netchunk)
                    image_pred = image_pred.squeeze()
                    loss = {
                        "proj_psnr": get_psnr(projs_pred, projs),
                        "proj_ssim": get_ssim(projs_pred, projs),
                        "psnr_3d": get_psnr_3d(image_pred, image),
                        "ssim_3d": get_ssim_3d(image_pred, image),
                    }










                    self.logger.info(loss)

                    resdir = os.path.join(self.evaldir,'the_last_eval')  
                    proj_pred_dir = os.path.join(resdir, "proj_pred")
                    proj_gt_dir = os.path.join(resdir, "proj_gt")

                    ct_pred_dir_H = os.path.join(resdir, "CT", "H", "ct_pred")
                    ct_gt_dir_H = os.path.join(resdir, "CT", "H", "ct_gt")
                    ct_pred_dir_W = os.path.join(resdir, "CT", "W", "ct_pred")
                    ct_gt_dir_W = os.path.join(resdir, "CT", "W", "ct_gt")
                    ct_pred_dir_L = os.path.join(resdir, "CT", "L", "ct_pred")
                    ct_gt_dir_L = os.path.join(resdir, "CT", "L", "ct_gt")

                    H, W, L = image_pred.shape
                    self.logger.info(image_pred.shape)

                    os.makedirs(proj_pred_dir, exist_ok=True)
                    os.makedirs(proj_gt_dir, exist_ok=True)
                    os.makedirs(ct_pred_dir_H, exist_ok=True)
                    os.makedirs(ct_gt_dir_H, exist_ok=True)
                    os.makedirs(ct_pred_dir_W, exist_ok=True)
                    os.makedirs(ct_gt_dir_W, exist_ok=True)
                    os.makedirs(ct_pred_dir_L, exist_ok=True)
                    os.makedirs(ct_gt_dir_L, exist_ok=True)

                    for i in tqdm(range(N)):
                        iio.imwrite(os.path.join(proj_pred_dir, f"proj_pred_{str(i)}.png"), ((cast_to_image(projs_pred[i]))*255).astype(np.uint8))
                        iio.imwrite(os.path.join(proj_gt_dir, f"proj_gt_{str(i)}.png"), ((cast_to_image(projs[i]))*255).astype(np.uint8))
            
                    for i in tqdm(range(H)):
                        iio.imwrite(os.path.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

                    for i in tqdm(range(W)):
                        iio.imwrite(os.path.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

                    for i in tqdm(range(L)):
                        iio.imwrite(os.path.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
                        iio.imwrite(os.path.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))
                self.net.train()
                if self.conf["render"]["n_fine"] > 0:
                    self.net_fine.train()
                npy = os.path.join(resdir, 'npy')
                os.makedirs(npy, exist_ok=True)
                np.save(os.path.join(npy, 'pred.npy'), projs_pred.cpu().numpy())
                np.save(os.path.join(npy, 'label.npy'), projs.cpu().numpy())

                np.save(os.path.join(npy, 'image_pred.npy'), image_pred.cpu().numpy())
                np.save(os.path.join(npy, 'image_abel.npy'), image.cpu().numpy())

                return projs_pred.cpu().numpy(),projs.cpu().numpy()


            def diffiusion_refine(self,pre_proj):  
                pre_proj , params = normalize_samples_global(pre_proj,self.conf)
                self.diffiusion_model.load()
                self.logger.info("--------------------------------------------------")
                self.logger.info("-----新角度细化------")
                self.logger.info("--------------------------------------------------")
                refine_proj = self.diffiusion_model.test(last=True,iter_data=pre_proj)
                refine_proj  = denormalize_samples_global(refine_proj,params,self.conf)
                torch.cuda.empty_cache() 

                self.train_dset.update_train(refine_proj)
                self.logger.info("--------------------------------------------------")
                self.logger.info("-----训练集角度更新------")
                self.logger.info(f"-----    角度数   -----  {len(self.train_dset.angles)}")
                self.logger.info("--------------------------------------------------")
                self.logger.info(f"{self.train_dset.angles}")




            def fmt_loss_str(self,losses):
                return "".join(", " + k + ": " + f"{losses[k].item():.4g}" for k in losses)


            def compute_loss(self, data):
                rays = data["rays"].reshape(-1, 8)  
                projs = data["projs"].reshape(-1)  
                ret = render(rays, self.net, self.net_fine,  ** self.conf["render"]) 
                projs_pred = ret["acc"]  
                loss = {"loss": 0.}
                calc_mse_loss(loss, projs, projs_pred)
                return loss["loss"]

            def eval_step(self, idx_epoch):




                image = self.train_dset.image
                image_pred = run_network(self.train_dset.voxels,
                                         self.net_fine if self.net_fine else self.net,
                                         self.conf["render"]["netchunk"])
                image_pred = image_pred.squeeze()  
                loss = {
                        "psnr_3d": get_psnr_3d(image_pred, image),
                        "ssim_3d": get_ssim_3d(image_pred, image),
                }




                if loss["psnr_3d"] > self.best_psnr_3d:
                    torch.save({
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.conf["render"]["n_fine"] else None,
                        "optimizer": self.optimizer.state_dict(),
                    }, self.ckpt_best_dir)
                    self.best_psnr_3d = loss["psnr_3d"]
                    self.logger.info(f"最佳模型更新，轮次:{idx_epoch}, 最佳3D PSNR:{self.best_psnr_3d:.4g}")

                resdir = os.path.join(self.evaldir, f"epoch_{idx_epoch:05d}")
                proj_pred_dir = os.path.join(resdir, "proj_pred")
                proj_gt_dir = os.path.join(resdir, "proj_gt")

                ct_pred_dir_H = os.path.join(resdir, "CT", "H", "ct_pred")
                ct_gt_dir_H = os.path.join(resdir, "CT", "H", "ct_gt")
                ct_pred_dir_W = os.path.join(resdir, "CT", "W", "ct_pred")
                ct_gt_dir_W = os.path.join(resdir, "CT", "W", "ct_gt")
                ct_pred_dir_L = os.path.join(resdir, "CT", "L", "ct_pred")
                ct_gt_dir_L = os.path.join(resdir, "CT", "L", "ct_gt")

                H, W, L = image_pred.shape
                self.logger.info(image_pred.shape)

                os.makedirs(proj_pred_dir, exist_ok=True)
                os.makedirs(proj_gt_dir, exist_ok=True)
                os.makedirs(ct_pred_dir_H, exist_ok=True)
                os.makedirs(ct_gt_dir_H, exist_ok=True)
                os.makedirs(ct_pred_dir_W, exist_ok=True)
                os.makedirs(ct_gt_dir_W, exist_ok=True)
                os.makedirs(ct_pred_dir_L, exist_ok=True)
                os.makedirs(ct_gt_dir_L, exist_ok=True)

                for i in tqdm(range(H)):
                    iio.imwrite(os.path.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

                for i in tqdm(range(W)):
                    iio.imwrite(os.path.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

                for i in tqdm(range(L)):
                    iio.imwrite(os.path.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
                    iio.imwrite(os.path.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))

                npy = os.path.join(resdir, 'npy')
                os.makedirs(npy, exist_ok=True)

                np.save(os.path.join(npy, 'image_pred.npy'), image_pred.cpu().numpy())
                np.save(os.path.join(npy, 'image_abel.npy'), image.cpu().numpy())
        
                return loss
    
            def update(self):
                if hasattr(self, 'logger'):
                    for handler in self.logger.handlers[:]:
                        handler.close()
                        self.logger.removeHandler(handler)
                self.iter_num += 1
                self.rootpath = os.path.join('logs','Iteration',  self.date_time,'iter_num_{}'.format(self.iter_num))
                self.expdir = os.path.join(self.rootpath,'Nerf_model_save')
                self.ckptdir = os.path.join(self.expdir, "ckpt.tar")
                self.ckptdir_backup = os.path.join(self.expdir, "ckpt_backup.tar")
                self.ckpt_best_dir = os.path.join(self.expdir, "ckpt_best.tar")
                self.evaldir = os.path.join(self.rootpath, "Nerf_eval")
                self.diffiusiondir = os.path.join(self.rootpath, "diffiusion")
                os.makedirs(self.expdir, exist_ok=True)
                os.makedirs(self.evaldir, exist_ok=True)
                os.makedirs(self.diffiusiondir, exist_ok=True)
                self.logger = gen_log(self.rootpath)
                self.best_psnr_3d = 0
            def load_nerf(self):
                ckpt = torch.load(self.conf["test"]["pre_model"],map_location=device)
                self.net.load_state_dict(ckpt["network"])
                self.net.eval()
                if self.conf["render"]["n_fine"] > 0:
                    self.net_fine.eval()

        

            def DR_to_sin(self,pre_proj):

                self.logger.info("--------------------------------------------------")
                self.logger.info("-----更新角度------")
                self.logger.info(f"-----角度------  {len(self.train_dset.angles)}")
                self.logger.info(f"{self.train_dset.angles}")
                self.logger.info("--------------------------------------------------")
                self.train_dset.update_train(pre_proj[0])

                projs = self.train_dset.label_projs  
                angle = self.train_dset.angles
                projs =  projs.permute(1, 0, 2)
                np.save(os.path.join('logs/temp', 'pred.npy'), projs.cpu().numpy())








        if __name__ == "__main__":
            cfg = {**load_config(args.config1),**load_config(args.config2)}
            device = torch.device(f"cuda:0")
            cfg['device'] = device
            cfg['phase'] = args.phase
            trainer = iterator(cfg)
            for i in range(args.cycle):
                trainer.Nerf_train()
                iter_data = trainer.Nerf_output(i+2)
                trainer.diffiusion_refine(iter_data)
                trainer.update()











            iter_data = trainer.Nerf_output(2)
            torch.cuda.empty_cache() 
