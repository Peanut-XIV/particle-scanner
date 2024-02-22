import os
import click
import datetime as dt
from pathlib import Path
from sashimi.controller import Controller
from sashimi import helicon_stack as helicon_stacker, utils, focus_stack


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dir', '-d', 'dir_',
              type=str,
              prompt='Directory to save images',
              default=None,
              help='path to a directory with a structure like "THIS/DIRECTORY/stacks/images.jpg"\n'
                   'With ')
@click.option('--port', '-p',
              type=str,
              prompt='COM port of printer',
              default='COM5',
              help='COM port of the 3D printer')
@click.option('--lang', '-l',
              type=str,
              default="en",
              prompt="Language",
              help='Language of the interface (en/fr)')
@click.option('--layout',
              type=str,
              default='AZERTY',
              help='Layout of the keyboard (QWERTY/AZERTY)')
@click.option("--mult-exp", "-e",
              type=str,
              is_flag=True,
              flag_value="undisclosed",
              default=None,
              help="Allows to use multiple Exposures")
@click.option("--remove-raw", "-r",
              is_flag=True,
              help='Removes the non-stacked pictures after finishing stacking')
@click.option("--skip-fs", "-s",
              is_flag=True,
              help='skips focus-stacking while scanning')
@click.option('--auto-quit', '-q',
              is_flag=True,
              help='Sashimi quits automatically after scanning')
@click.option('--margin', '-m',
              type=int,
              default=200,
              help="the subtracted margin to a stack's lowest point")
@click.option('--lowest', '-z',
              is_flag=True,
              help='simplifies z correction')
@click.option('--yes/--no', '-y/-n',
              default=False,
              is_flag=True)
def scan(dir_, port, lang, layout, mult_exp, remove_raw, skip_fs, auto_quit, margin, lowest, yes):
    if dir_ is None:
        dir_ = utils.make_unique_subdir()
    if mult_exp == 'undisclosed':
        exp_values, dir_ = dialog_for_path_and_values()
    elif mult_exp is not None:
        exp_values = mult_exp.split(",").map(lambda x: int((x.strip(' '))))
    else:
        exp_values = None
        
    controller = Controller(dir_, port, lang=lang, layout=layout,
                            z_margin=margin, remove_raw=remove_raw,
                            auto_f_stack=not skip_fs, auto_quit=auto_quit,
                            multi_exp=exp_values, lowest_z=lowest, do_overwrite=yes)
    controller.start()


@cli.command()
@click.option('--dir', '-d', 'dir_',
              type=str,
              prompt='Directory containing stacks',
              help='Directory with a structure like "THIS/DIRECTORY/X00100_Y02200/X00100_Y02200_Z00135.jpg"')
def stack(dir_):
    if dir_ is None:
        utils.make_unique_subdir()
    else:
        dir_ = Path(dir_).resolve()
    focus_stack.stack(dir_)
    print(f"focus stacks saved at {dir_}")


@cli.command()
@click.option('--dir', '-d', 'dir_',
              type=str,
              prompt='Directory containing stacks',
              default=None,
              help='the input directory from which pictures will be stacked.\n'
                   'It must have a structure like "THIS/DIRECTORY/image_stack/image.jpg"')
@click.option('--output-dir', '-o',
              type=str,
              default=None,
              help='The directory in which focus stacks will be saved.\n'
                   'When not specified, it the same a the input directory.')
@click.option('-y/-n', '--exists-ok', default=False,
              help='makes the command error out if the output dir already exists.')
def helicon_stack(dir_, output_dir, exists_ok):
    dir_ = Path(dir_)
    depth = get_homogeneous_depth(dir_)
    
    if depth == 2 or depth == 3:
        if output_dir is None:
            output_dir = dir_.joinpath("f_stacks")
        else:
            output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=exists_ok)
    
    if depth == 3:
        xy_folder = next(dir_.iterdir())
        exposures = sorted([int(folder.stem.lstrip()) for folder in xy_folder.iterdir()])
        helicon_stacker.stack_for_multiple_exp(dir_, output_dir, exposures)
        print(f"focus stacks saved at {output_dir}")
    elif depth == 2:
        helicon_stacker.stack_from_to(dir_, dir)
        print(f"focus stacks saved at {output_dir}")
    else:
        print('ERROR: incompatible directory')


def get_homogeneous_depth(path):
    if path.is_file() or path is None:
        return 0
    child = next(path.iterdir())
    return get_homogeneous_depth(child) + 1


def dialog_for_path_and_values() -> (Path, list[int]):
    u_path = utils.ask_for_path()
    interval = utils.ask_for_interval()
    step_nbr = utils.ask_for_step_nbr(interval)
    step_size = (interval[1] - interval[0]) // (step_nbr - 1)
    exp_val = [interval[0] + i * step_size for i in range(step_nbr)]
    return u_path, exp_val


if __name__ == "__main__":
    cli()
