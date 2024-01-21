import torch
import numpy as np
from im2scene.common import interpolate_sphere
from torchvision.utils import save_image, make_grid
import imageio
from math import sqrt
from os import makedirs
from os.path import join


class Renderer(object):
    '''  Render class for GIRAFFE.

    It provides functions to render the representation.

    Args:
        model (nn.Module): trained GIRAFFE model
        device (device): pytorch device
    '''

    def __init__(self, model, device=None, mode = None):
        self.model = model.to(device)
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        iden = self.model.identifier
        iden.eval()
        self.identifier = iden
        self.generator = gen
        self.device = device
        self.mode = mode
        self.render_batch_size = gen.render_batch_size

        # sample temperature; only used for visualiations
        self.sample_tmp = 0.65

        data = next(iter(gen.render_loader))
        self.cond_data = data['cond'].to(self.device)
        self.image_size = data['cond'].shape[-1]

    def set_random_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def render_full_visualization(self, img_out_path,
                                  render_program=['object_rotation']):

        batch_size = self.render_batch_size
        for rp in render_program:
            if rp == 'object_rotation':
                self.set_random_seed()
                self.render_object_rotation(img_out_path, batch_size = batch_size)
            if rp == 'object_translation_horizontal':
                self.set_random_seed()
                self.render_object_translation_horizontal(img_out_path, batch_size = batch_size)
            if rp == 'object_translation_vertical':
                self.set_random_seed()
                self.render_object_translation_depth(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_app':
                self.set_random_seed()
                self.render_interpolation(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_app_bg':
                self.set_random_seed()
                self.render_interpolation_bg(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_shape':
                self.set_random_seed()
                self.render_interpolation(img_out_path, batch_size = batch_size, mode='shape')
            if rp == 'render_camera_elevation':
                self.set_random_seed()
                self.render_camera_elevation(img_out_path, batch_size = batch_size)
            if rp == 'render_add_cars':
                self.set_random_seed()
                self.render_add_objects_cars5(img_out_path, batch_size = batch_size)

    def render_full_inversion_visualization(self, img_out_path,
                                  render_program=['object_rotation']):
        batch_size = self.render_batch_size
        for rp in render_program:
            if rp == 'object_rotation':
                self.set_random_seed()
                self.render_object_rotation_inversion(img_out_path, batch_size = batch_size)
            if rp == 'object_translation_horizontal':
                self.set_random_seed()
                self.render_object_translation_horizontal_inversion(img_out_path, batch_size = batch_size)
            if rp == 'object_translation_vertical':
                self.set_random_seed()
                self.render_object_translation_depth_inversion(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_app':
                self.set_random_seed()
                self.render_interpolation_inversion(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_app_bg':
                self.set_random_seed()
                self.render_interpolation_bg_inversion(img_out_path, batch_size = batch_size)
            if rp == 'interpolate_shape':
                self.set_random_seed()
                self.render_interpolation_inversion(img_out_path, batch_size = batch_size, mode='shape')
            if rp == 'render_camera_elevation':
                self.set_random_seed()
                self.render_camera_elevation_inversion(img_out_path, batch_size = batch_size)
            if rp == 'render_add_cars':
                self.set_random_seed()
                self.render_add_objects_cars5_inversion(img_out_path, batch_size = batch_size)
                
    
    
    def render_object_rotation(self, img_out_path, batch_size=2, n_steps=32):
        gen = self.generator
        bbox_generator = gen.bounding_box_generator

        n_boxes = bbox_generator.n_boxes

        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]
        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = []
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)
            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'rotation_object')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation))
        
    
    def render_object_translation_horizontal(self, img_out_path, batch_size=2, n_steps=32):
        gen = self.generator

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[x_val, i, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'translation_object_horizontal')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_horizontal',
            add_reverse=True)        
          
    def render_object_translation_depth(self, img_out_path, batch_size=2, n_steps=32):
        gen = self.generator
        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            y_val = 0.5
        elif n_boxes == 2:
            t = [[0.4, 0.8, 0.]]
            y_val = 0.2

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[i, y_val, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_depth')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_depth', add_reverse=True)

    def render_interpolation(self, img_out_path, batch_size=2, n_samples=6,
                             n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_obj_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]
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

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_ii, z_shape_bg_1,
                                    z_app_bg_1]
                else:
                    latent_codes = [z_ii, z_app_obj_1, z_shape_bg_1,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation,  mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True)

    def render_interpolation_bg(self, img_out_path, batch_size=2, n_samples=6,
                                n_steps=32, mode='app'):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = \
            gen.get_latent_codes(batch_size, tmp=self.sample_tmp)

        z_i = [
            gen.sample_z(
                z_app_bg_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

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

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_shape_bg_1,
                                    z_ii]
                else:
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_ii,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation,  mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_bg_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_bg_%s' % mode,
            is_full_rotation=True)


    def render_camera_elevation(self, img_out_path, batch_size=2, n_steps=32):
        gen = self.generator
        n_boxes = gen.bounding_box_generator.n_boxes
        r_range = [0.1, 0.9]

        # Get values
        latent_codes = gen.get_latent_codes(batch_size, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            [0.5 for i in range(n_boxes)],
            batch_size,
        )

        out = []
        for step in range(n_steps):
            v = step * 1.0 / (n_steps - 1)
            r = r_range[0] + v * (r_range[1] - r_range[0])
            camera_matrices = gen.get_camera(val_v=r, batch_size=batch_size)
            with torch.no_grad():
                out_i = gen(
                    batch_size, latent_codes, camera_matrices, transformations, bg_rotation, 
                    mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'camera_elevation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='elevation_camera',
                                   is_full_rotation=False)

    def render_add_objects_cars5(self, img_out_path, batch_size=2):

        gen = self.generator

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
        ]

        t = [
            [-0.7, -.8, 0.],
            [-0.7, 0.5, 0.],
            [-0.7, 1.8, 0.],
            [1.5, -.8, 0.],
            [1.5, 0.5, 0.],
            [1.5, 1.8, 0.],
        ]
        r = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        outs = []
        for i in range(1, 7):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            outs.append(out)
        outs = torch.stack(outs)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = outs[[idx]]

        # import pdb; pdb.set_trace()
        out_folder = join(img_out_path, 'add_cars')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_cars',
                                   is_full_rotation=False, add_reverse=True)
        

    def render_test(self, img_out_path,
                                  render_program=['object_rotation']):
        batch_size = self.render_batch_size
        gen = self.generator
        data_loader = gen.render_loader
        for i, cond in enumerate(data_loader):
            print("idx:",i)
            for rp in render_program:
                if rp == 'object_rotation':
                    self.set_random_seed()
                    self.render_object_rotation_inversion(img_out_path, cond, i,  batch_size = batch_size)
                if rp == 'object_translation_horizontal':
                    self.set_random_seed()
                    self.render_object_translation_horizontal_inversion(img_out_path, batch_size = batch_size)
                if rp == 'object_translation_vertical':
                    self.set_random_seed()
                    self.render_object_translation_depth_inversion(img_out_path, batch_size = batch_size)
                if rp == 'interpolate_app':
                    self.set_random_seed()
                    self.render_interpolation_inversion(img_out_path, cond, i, batch_size = batch_size)
                if rp == 'interpolate_app_bg':
                    self.set_random_seed()
                    self.render_interpolation_bg_inversion(img_out_path, batch_size = batch_size)
                if rp == 'interpolate_shape':
                    self.set_random_seed()
                    self.render_interpolation_inversion(img_out_path, cond, i, mode='shape', batch_size = batch_size)
                if rp == 'render_camera_elevation':
                    self.set_random_seed()
                    self.render_camera_elevation_inversion(img_out_path, batch_size = batch_size)
                if rp == 'render_add_cars':
                    self.set_random_seed()
                    self.render_add_objects_cars5_inversion(img_out_path, batch_size = batch_size)
                if rp == 'render_add_clevr10':
                    self.set_random_seed()
                    self.render_add_objects_clevr10_inversion(img_out_path, batch_size = batch_size)
                if rp == 'render_add_clevr6':
                    self.set_random_seed()
                    self.render_add_objects_clevr6_inversion(img_out_path, batch_size = batch_size)


    def render_object_rotation_inversion(self, img_out_path, cond_data = None , batch_idx = 0, batch_size=2, n_steps=32):
        gen = self.generator
        iden = self.identifier
        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes

        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        latent_codes = []
        # Get Random codes and bg rotation
        if self.mode == 'test':
            cond_data = cond_data.get('cond').to(self.device)
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            cond_data = self.cond_data
            latent_codes = iden(cond_data, batch_size = batch_size)
        bg_rotation = gen.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]
        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = [cond_data.cpu()]
        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'rotation_object')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation), batch_idx = batch_idx)

    
    def render_object_translation_horizontal_inversion(self, img_out_path, cond_data = None , batch_size=2,
                                             n_steps=32):
        gen = self.generator

        # Get values
        if self.mode == 'test':
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            latent_codes = iden(self.cond_data, batch_size = batch_size)

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)
        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            x_val = 0.5
        elif n_boxes == 2:
            t = [[0.5, 0.5, 0.]]
            x_val = 1.

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[x_val, i, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'translation_object_horizontal')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_horizontal',
            add_reverse=True)        
          
    def render_object_translation_depth_inversion(self, img_out_path, cond_data = None , batch_size=2,
                                        n_steps=32):
        gen = self.generator
        iden = self.identifier
        # Get values
        if self.mode == 'test':
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            latent_codes = iden(self.cond_data, batch_size = batch_size)

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(batch_size=batch_size)

        n_boxes = gen.bounding_box_generator.n_boxes
        s = [[0., 0., 0.]
             for i in range(n_boxes)]
        r = [0.5 for i in range(n_boxes)]

        if n_boxes == 1:
            t = []
            y_val = 0.5
        elif n_boxes == 2:
            t = [[0.4, 0.8, 0.]]
            y_val = 0.2

        out = []
        for step in range(n_steps):
            i = step * 1.0 / (n_steps - 1)
            ti = t + [[i, y_val, 0.]]
            transformations = gen.get_transformations(s, ti, r, batch_size)
            with torch.no_grad():
                out_i = gen(batch_size, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'translation_object_depth')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='translation_depth', add_reverse=True)

    def render_interpolation_inversion(self, img_out_path, cond_data = None , batch_idx = 0, batch_size=2, n_samples=6,
                             n_steps=32, mode='app'):
        gen = self.generator
        iden = self.identifier
        n_boxes = gen.bounding_box_generator.n_boxes

        # Get values
        if self.mode == 'test':
            cond_data = cond_data.get('image').to(self.device)
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            latent_codes = iden(self.cond_data, batch_size = batch_size)
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = latent_codes
        z_i = [
            gen.sample_z(
                z_app_obj_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

        

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

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_ii, z_shape_bg_1,
                                    z_app_bg_1]
                else:
                    latent_codes = [z_ii, z_app_obj_1, z_shape_bg_1,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation,  mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_%s' % mode,
            is_full_rotation=True, batch_idx = batch_idx)

    def render_interpolation_bg_inversion(self, img_out_path, cond_data = None , batch_size=2, n_samples=6,
                                n_steps=32, mode='app'):
        gen = self.generator
        iden = self.identifier

        n_boxes = gen.bounding_box_generator.n_boxes
        # Get values
        if self.mode == 'test':
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            latent_codes = iden(self.cond_data, batch_size = batch_size)
        z_shape_obj_1, z_app_obj_1, z_shape_bg_1, z_app_bg_1 = latent_codes
        z_i = [
            gen.sample_z(
                z_app_bg_1.shape,
                tmp=self.sample_tmp) for j in range(n_samples)
        ]

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

        out = []
        for j in range(n_samples):
            z_i1 = z_i[j]
            z_i2 = z_i[(j+1) % (n_samples)]
            for step in range(n_steps):
                w = step * 1.0 / ((n_steps) - 1)
                z_ii = interpolate_sphere(z_i1, z_i2, w)
                if mode == 'app':
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_shape_bg_1,
                                    z_ii]
                else:
                    latent_codes = [z_shape_obj_1, z_app_obj_1, z_ii,
                                    z_app_bg_1]
                with torch.no_grad():
                    out_i = gen(batch_size, latent_codes, camera_matrices,
                                transformations, bg_rotation,  mode='val')
                out.append(out_i.cpu())
        out = torch.stack(out)

        # Save Video
        out_folder = join(img_out_path, 'interpolate_bg_%s' % mode)
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(
            out, out_folder, name='interpolate_bg_%s' % mode,
            is_full_rotation=True)


    def render_camera_elevation_inversion(self, img_out_path, cond_data = None , batch_size=2, n_steps=32):
        gen = self.generator
        iden = self.identifier
        n_boxes = gen.bounding_box_generator.n_boxes
        r_range = [0.1, 0.9]

        # Get values
        if self.mode == 'test':
            latent_codes = iden(cond_data, batch_size = batch_size)
        else:
            latent_codes = iden(self.cond_data, batch_size = batch_size)

        bg_rotation = gen.get_random_bg_rotation(batch_size)
        transformations = gen.get_transformations(
            [[0., 0., 0.] for i in range(n_boxes)],
            [[0.5, 0.5, 0.5] for i in range(n_boxes)],
            [0.5 for i in range(n_boxes)],
            batch_size,
        )

        out = []
        for step in range(n_steps):
            v = step * 1.0 / (n_steps - 1)
            r = r_range[0] + v * (r_range[1] - r_range[0])
            camera_matrices = gen.get_camera(val_v=r, batch_size=batch_size)
            with torch.no_grad():
                out_i = gen(
                    batch_size, latent_codes, camera_matrices, transformations, bg_rotation, 
                    mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)

        out_folder = join(img_out_path, 'camera_elevation')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(out, out_folder, name='elevation_camera',
                                   is_full_rotation=False)

    def render_add_objects_cars5_inversion(self, img_out_path, batch_size=2):

        gen = self.generator

        # Get values
        z_shape_obj, z_app_obj, z_shape_bg, z_app_bg = gen.get_latent_codes(
            batch_size, tmp=self.sample_tmp)
        z_shape_obj = gen.sample_z(
            z_shape_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        z_app_obj = gen.sample_z(
            z_app_obj[:, :1].repeat(1, 6, 1).shape, tmp=self.sample_tmp)
        bg_rotation = gen.get_random_bg_rotation(batch_size)
        camera_matrices = gen.get_camera(val_v=0., batch_size=batch_size)

        s = [
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.],
        ]

        t = [
            [-0.7, -.8, 0.],
            [-0.7, 0.5, 0.],
            [-0.7, 1.8, 0.],
            [1.5, -.8, 0.],
            [1.5, 0.5, 0.],
            [1.5, 1.8, 0.],
        ]
        r = [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
        outs = []
        for i in range(1, 7):
            transformations = gen.get_transformations(
                s[:i], t[:i], r[:i], batch_size)
            latent_codes = [z_shape_obj[:, :i], z_app_obj[:, :i], z_shape_bg,
                            z_app_bg]
            with torch.no_grad():
                out = gen(
                    batch_size, latent_codes, camera_matrices, transformations,
                    bg_rotation, mode='val').cpu()
            outs.append(out)
        outs = torch.stack(outs)
        idx = torch.arange(6).reshape(-1, 1).repeat(1, (128 // 6)).reshape(-1)
        outs = outs[[idx]]

        # import pdb; pdb.set_trace()
        out_folder = join(img_out_path, 'add_cars')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images(outs, out_folder, name='add_cars',
                                   is_full_rotation=False, add_reverse=True)

    
    # Helper functions
    def write_video(self, out_file, img_list, n_row=5, add_reverse=False
                    ):
        n_steps, batch_size = img_list.shape[:2]
        nrow = n_row if (n_row is not None) else int(sqrt(batch_size))
        img = [(255*make_grid(img, nrow=nrow, pad_value=1.).permute(
            1, 2, 0)).cpu().numpy().astype(np.uint8) for img in img_list]
        if add_reverse:
            img += list(reversed(img))
        imageio.mimwrite(out_file, img, fps=30, quality=8)

    def save_video_and_images(self, imgs, out_folder, name='rotation_object',
                              is_full_rotation=False, img_n_steps=6,
                              add_reverse=False, batch_idx = 0):
        img_size = self.image_size
        cond = imgs[0]      
        imgs = imgs[1:]    

        # Save video
        out_file_video = join(out_folder, '%s%03d.mp4' % (name, batch_idx))
        self.write_video(out_file_video, imgs, add_reverse=add_reverse)

        # Save images
        n_steps, batch_size = imgs.shape[:2]
        if is_full_rotation:
            idx_paper = np.linspace(
                0, n_steps - n_steps // img_n_steps, img_n_steps
            ).astype(np.int)
        else:
            idx_paper = np.linspace(0, n_steps - 1, img_n_steps).astype(np.int)
        for idx in range(batch_size):
            img_grid = torch.cat([cond[idx].reshape(1,3,img_size,img_size),imgs[idx_paper, idx]], dim = 0)    
            save_image(make_grid(
                img_grid, nrow=img_n_steps+1, pad_value=1.), join(
                    out_folder, '%05d_%s.jpg' % (batch_idx*batch_size + idx, name)))


#
    def render_interpolate(self, img_out_path,
                                  render_program=['object_rotation']):

        batch_size = 1
        gen = self.generator
        data_loader = gen.render_loader
        for i, cond in enumerate(data_loader):
            print("idx:",i)
            for rp in render_program:
                if rp == 'object_rotation':
                    self.set_random_seed()
                    self.render_object_rotation_interpolate(img_out_path,cond_data= cond, batch_idx = i, batch_size = batch_size)
         


    def render_object_rotation_interpolate(self, img_out_path, cond_data = None , batch_idx = 0, batch_size=1, n_steps=32):
        gen = self.generator
        iden = self.identifier
        bbox_generator = gen.bounding_box_generator
        n_boxes = bbox_generator.n_boxes

        cond_data = cond_data.get('image').to(self.device)
        image_size = cond_data.shape[-1]

        # Set rotation range
        is_full_rotation = (bbox_generator.rotation_range[0] == 0
                            and bbox_generator.rotation_range[1] == 1)
        n_steps = int(n_steps * 2) if is_full_rotation else n_steps
        r_scale = [0., 1.] if is_full_rotation else [0.1, 0.9]

        # Get Random codes and bg rotation
        z_shape_obj, _, z_shape_bg, z_app_bg = iden(cond_data[0].reshape(1, 3, image_size, image_size), batch_size = 1)
        _ , z_app_obj, _ , _ = iden(cond_data[1].reshape(1, 3, image_size, image_size), batch_size = 1)
        latent_codes = [z_shape_obj, z_app_obj, z_shape_bg, z_app_bg]
        bg_rotation = gen.get_random_bg_rotation(batch_size)

        # Set Camera
        camera_matrices = gen.get_camera(batch_size=batch_size)
        s_val = [[0, 0, 0] for i in range(n_boxes)]
        t_val = [[0.5, 0.5, 0.5] for i in range(n_boxes)]
        r_val = [0. for i in range(n_boxes)]
        s, t, _ = gen.get_transformations(s_val, t_val, r_val, batch_size)

        out = [cond_data[0].reshape(1,3,image_size,image_size).cpu(), cond_data[1].reshape(1, 3, image_size, image_size).cpu()]

        for step in range(n_steps):
            # Get rotation for this step
            r = [step * 1.0 / (n_steps - 1) for i in range(n_boxes)]
            r = [r_scale[0] + ri * (r_scale[1] - r_scale[0]) for ri in r]
            r = gen.get_rotation(r, batch_size)

            # define full transformation and evaluate model
            transformations = [s, t, r]
            with torch.no_grad():
                out_i = gen(1, latent_codes, camera_matrices,
                            transformations, bg_rotation,  mode='val')
            out.append(out_i.cpu())
        out = torch.stack(out)
        out_folder = join(img_out_path, 'rotation_object')
        makedirs(out_folder, exist_ok=True)
        self.save_video_and_images_interpolate(
            out, out_folder, name='rotation_object',
            is_full_rotation=is_full_rotation,
            add_reverse=(not is_full_rotation), batch_idx = batch_idx)

    def save_video_and_images_interpolate(self, imgs, out_folder, name='rotation_object',
                              is_full_rotation=False, img_n_steps=6,
                              add_reverse=False, batch_idx = 0):
        img_size = self.image_size
        cond = imgs[:2]      
        imgs = imgs[2:]    

        # Save video
        out_file_video = join(out_folder, '%s%03d.mp4' % (name, batch_idx))
        self.write_video(out_file_video, imgs, add_reverse=add_reverse)
        # Save images
        n_steps, batch_size = imgs.shape[:2]
        if is_full_rotation:
            idx_paper = np.linspace(
                0, n_steps - n_steps // img_n_steps, img_n_steps
            ).astype(np.int)
        else:
            idx_paper = np.linspace(0, n_steps - 1, img_n_steps).astype(np.int)
        for idx in range(batch_size):
            img_grid = torch.cat([cond[0].reshape(1,3,img_size,img_size),cond[1].reshape(1,3,img_size,img_size),imgs[idx_paper, idx]], dim = 0)    
            save_image(make_grid(
                img_grid, nrow=img_n_steps+2, pad_value=1.), join(
                    out_folder, '%05d_%s.jpg' % (batch_idx*batch_size + idx, name)))
