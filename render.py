import torch
import os
import argparse
from im2scene import config
from im2scene.checkpoints import CheckpointIO

parser = argparse.ArgumentParser(
    description='Render images of a GIRAFFE model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
out_dir = cfg['training']['out_dir']
render_dir = os.path.join(out_dir, cfg['rendering']['render_dir'])
render_inv_dir = os.path.join(out_dir, cfg['rendering']['render_inv_dir'])
test_dir = os.path.join(out_dir, cfg['rendering']['test_dir'])
interpolate_dir = os.path.join(out_dir, cfg['rendering']['interpolate_dir'])

# Model
model = config.get_model(cfg, device=device)
checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg['test']['model_file'])
# Generator
renderer = config.get_renderer(model, cfg, device=device)

model.eval()

if cfg['rendering']['mode']=='train':
    out = renderer.render_full_visualization(
        render_dir,
        cfg['rendering']['render_program'])

elif cfg['rendering']['mode']=='inversion':
    out_inversion = renderer.render_full_inversion_visualization(
        render_inv_dir,
        cfg['rendering']['render_program'])

elif cfg['rendering']['mode']=='test':
    out_inversion = renderer.render_test(
        test_dir,
        cfg['rendering']['render_program'])

elif cfg['rendering']['mode'] == 'interpolate':
    out_inversion = renderer.render_interpolate(
        interpolate_dir,
        cfg['rendering']['render_program'])
