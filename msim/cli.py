import click
from msim.simulate import run_simulation, load_config
from msim.vis_volume import view_volume
import os

@click.group()
def main():
    pass

@main.command()
@click.argument("config", type=click.Path())
def run(config):
    """Run simulation using CONFIG file"""
    if not os.path.exists(config):
        click.echo(f"[INFO] Config file '{config}' not found. Creating default config.")
        load_config(config)  # this will create and return the default config
    run_simulation(config)

@main.command()
@click.argument("volume_path", type=click.Path(exists=True))
def vis(volume_path):
    """Visualize a volume (.n5 or .zarr)"""
    view_volume(volume_path)
