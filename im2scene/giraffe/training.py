from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from im2scene.losses import VGGPerceptualLoss, CombinationLoss
from im2scene import pytorch_ssim
from torchvision.utils import save_image, make_grid
import os
import torch
import torch.nn as nn
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)

import torch.optim as optim


class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, optimizer_d, optimizer_i, device=None,
                 vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True,
                 inv_start_iter = 200000,
                 **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.optimizer_i = optimizer_i
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations
        self.inv_start_iter = inv_start_iter

        self.vis_dict = model.generator.get_vis_dict(64)

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(self.model.discriminator)
            self.identifier = torch.nn.DataParallel(self.model.identifier)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test
            self.identifier = self.model.identifier

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        if it<=self.inv_start_iter:
            loss_g = self.train_step_generator(data, it)
            loss_d, reg_d, fake_d, real_d = self.train_step_discriminator(data, it)
            return{
                'G': loss_g,
                'D': loss_d,
                'reg': reg_d,                
            }
        else:
            loss_i, loss_latent,  loss_cycle, loss_reconst_adv, loss_ssim,loss_vgg = self.train_step_identifier(data, it)
            return {
                'I': loss_i,
                'I_latent' : loss_latent,
                'I_cycle' : loss_cycle,
                'I_reconst_adv' :loss_reconst_adv ,
                'I_ssim' : loss_ssim,
                'I_vgg' : loss_vgg
            }

            
    def eval_step(self):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        x_fake = []
        n_iter = self.n_eval_iterations

        for i in tqdm(range(n_iter)):
            with torch.no_grad():
                x_fake.append(gen().cpu()[:, :3])
        x_fake = torch.cat(x_fake, dim=0)
        x_fake.clamp_(0., 1.)
        mu, sigma = calculate_activation_statistics(x_fake)
        fid_score = calculate_frechet_distance(
            mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
        eval_dict = {
            'fid_score': fid_score
        }

        return eval_dict

    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)
        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        if self.multi_gpu:
            latents = generator.module.get_train_dict()
            x_fake = generator(**latents)
        else:
            x_fake = generator()

        d_fake = discriminator(x_fake)
        gloss = compute_bce(d_fake, 1)

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gloss.item()
            

    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        generator.train()
        discriminator.train()
        
        

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)

        loss_d_full = 0.

        x_real.requires_grad_()

        d_real = discriminator(x_real)

        d_loss_real = compute_bce(d_real, 1)
        loss_d_full += d_loss_real

        reg = 10. * compute_grad2(d_real, x_real).mean()
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_train_dict()
                x_fake = generator(**latents)
            else:
                x_fake = generator()

        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)

        d_loss_fake = compute_bce(d_fake, 0)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_fake + d_loss_real)

        return (
            d_loss.item(), reg.item(), d_loss_fake.item(), d_loss_real.item())
    
    def train_step_identifier(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        identifier = self.identifier
        toggle_grad(generator, False)
        toggle_grad(discriminator, False)
        toggle_grad(identifier, True)
        generator.train()
        discriminator.train()
        identifier.train()

        self.optimizer_i.zero_grad()

        gen_train_dict = generator.get_train_dict(32)
        
        
        ssim = pytorch_ssim.SSIM()
        vgg_perceptual = VGGPerceptualLoss().to(self.device)
        
        with torch.no_grad():
            x_fake = generator(**gen_train_dict)           
            
        x_fake.requires_grad_() 
        latents_predict  = identifier(x_fake, batch_size = 32)          

        reconst_dict = {
            'batch_size': gen_train_dict['batch_size'],
            'latent_codes': latents_predict,            
            'camera_matrices': gen_train_dict['camera_matrices'],
            'transformations': gen_train_dict['transformations'],
            'bg_rotation': gen_train_dict['bg_rotation'],
        }
        
        x_reconst = generator(**reconst_dict)
        d_reconst = discriminator(x_reconst)
        
        compute_l1 = nn.L1Loss()
                
        loss_latent_1 = compute_l1(gen_train_dict['latent_codes'][0],latents_predict[0])/2
        loss_latent_2 = compute_l1(gen_train_dict['latent_codes'][1],latents_predict[1])/2
        loss_latent_3 = compute_l1(gen_train_dict['latent_codes'][2],latents_predict[2])/8
        loss_latent_4 = compute_l1(gen_train_dict['latent_codes'][3],latents_predict[3])/8
        
        loss_latent = (loss_latent_1+ loss_latent_2+loss_latent_3+loss_latent_4)*10
        loss_cycle = compute_l1(x_fake, x_reconst)*10
        loss_reconst_adv = compute_bce(d_reconst ,1)
        loss_ssim = (1 - ssim(x_fake, x_reconst))
        loss_vgg = vgg_perceptual(x_fake, x_reconst)*0.1
        
        loss_i = loss_latent+ loss_cycle + loss_reconst_adv + loss_ssim + loss_vgg

        loss_i.backward()
        self.optimizer_i.step()

        return (
           loss_i.item(), 
           loss_latent.item(),  
           loss_cycle.item(), 
           loss_reconst_adv.item(),          
           loss_ssim.item(), 
           loss_vgg.item() ) 

    def optimization(self, data, it):
        generator = self.generator
        toggle_grad(generator, False)
        iter = 100
        GT = data.get('image').to(self.device)
        loss_function = CombinationLoss(self.device)

        variable_estimate = generator.get_optimize_dict(1)
        estimate_list = []

        for key, item in variable_estimate.items():
            for i in variable_estimate[key]:
                estimate_list.append(i)
        for i, variable in enumerate(estimate_list):
            variable.requires_grad = True
        optimizer_optim = optim.Adam(estimate_list, lr = 0.001)
        
        
        estimate_dict = {
            'batch_size': 1,
            'latent_codes': (estimate_list[0], estimate_list[1], estimate_list[2], estimate_list[3]),            
            'camera_matrices': (estimate_list[4], estimate_list[5]),
            'transformations': (estimate_list[6], estimate_list[7], estimate_list[8]),
            'bg_rotation': estimate_list[9],
        }

        for i in tqdm(range(iter)):
            x_estimate = generator(**estimate_dict)
            optimizer_optim.zero_grad()
            loss,l1, l2, vgg = loss_function(x_estimate, GT)
            loss.backward()
            optimizer_optim.step()
        
        generator.eval()
        with torch.no_grad():
            result = generator(**estimate_dict)

        out_file_name = str(it)+'.jpg'

        image_grid_GT = make_grid(GT.clamp_(0., 1.))
        save_image(image_grid_GT, os.path.join(self.vis_dir+'/GT', out_file_name))
        
        image_grid = make_grid(result.clamp_(0., 1.))
        save_image(image_grid, os.path.join(self.vis_dir+'/Fake', out_file_name))
        print("it: ", it) 

        return image_grid

    
    def eval_data(self, data, it):

        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        n_boxes = 1
        batch_size = 1
        GT = data.get('image').to(self.device)

        inversion_dict = {
            'x' : GT,
            'batch_size' : batch_size
        }

        latent = self.identifier(**inversion_dict)

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        if n_boxes == 1:
            t_val = [[0.5, 0.5, 0.5]]
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            t_val,
            [0.5 for i in range(n_boxes)],
            batch_size
        )

        reconst_dict = {
            'batch_size': batch_size,
            'camera_matrices': camera_matrices,
            'transformations': transformations,
            'bg_rotation': bg_rotation,
        }

        with torch.no_grad():
            image_fake = self.generator(**reconst_dict, latent_codes = latent, mode='val').cpu()

        
        out_file_name = str(it)+'.png'

        iamge_grid_GT = make_grid(image_fake.clamp_(0., 1.))
        save_image(GT, os.path.join('eval_data/afhq/128/GT', out_file_name))
        
        image_grid = make_grid(image_fake.clamp_(0., 1.))
        save_image(image_grid, os.path.join('eval_data/afhq/128/Fake', out_file_name))
        print("it: ", it) 

        
    def visualize(self, it=0):
        ''' Visualized the data.
        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            image_fake = self.generator(**self.vis_dict, mode='val').cpu()

        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it

        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=8)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid
    
    def visualize_inversion(self, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()

        inversion_dict = {
            'x' : self.vis_dict['cond_data'],
            'batch_size' : 64
        }

        latent = self.identifier(**inversion_dict)
        reconst_dict = {
            'batch_size': self.vis_dict['batch_size'],
            'camera_matrices': self.vis_dict['camera_matrices'],
            'transformations': self.vis_dict['transformations'],
            'bg_rotation': self.vis_dict['bg_rotation'],
        }
        with torch.no_grad():
            image_fake = self.generator(**reconst_dict, latent_codes = latent, mode='val').cpu()

        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d_inv.png' % it
        
        image_grid = make_grid(image_fake.clamp_(0., 1.), nrow=8)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid 
    
    def visualize_cond(self, it=0):
        ''' Visualized the conditional data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():            
            image_cond = self.vis_dict['cond_data'].cpu()

        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_cond.png'

        image_grid = make_grid(image_cond.clamp_(0., 1.), nrow=8)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid         
    
    
