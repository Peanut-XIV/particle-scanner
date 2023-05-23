import click
from sashimi import focus_stack
from sashimi import helicon_stack as helicon_stacker
from sashimi.controller import Controller
from sashimi.multi_exp import dialog_for_path_and_values


@click.group()
def cli():
    pass


@cli.command()
@click.argument(dcls=[None, '--dir', '-d', 'dir_'],
                type=str,
                default='~/images/sashimi/',
                prompt='Directory to save images',
                help='Directory to save images')
@click.argument(dcls=[None, '--port', '-p'],
                type=str,
                default='COM5',
                prompt='COM port of printer',
                help='COM port of the 3D printer')
@click.option(dcls=['--lang', '-l'],
              type=str,
              default="en",
              help='Language of the interface (en/fr)')
@click.option('--layout',
              type=str,
              default='QWERTY',
              help='Layout of the keyboard (QWERTY/AZERTY)')
@click.option(dcls=["--skipfs", "-s"],
              is_flag=True,
              flag_value=True,
              default=False,
              help='disable the automatic focus-stacking of pictures after a scan')
@click.option(dcls=['--autoquit', '-q'],
              is_flag=True,
              flag_value=True,
              default=False,
              help='sashimi quits automatically after scanning')
@click.option(dcls=['--offset', '-o'],
              type=int,
              default=1000,
              help='z offset in top-down mode')
@click.option(dcls=['--lowest', '-z'],
              is_flag=True,
              flag_value=True,
              default=False,
              help='simplifies z correction')
def scan(dir_, port, lang, layout, skipfs, autoquit, offset, lowest):
    controller = Controller(dir_,
                            port,
                            lang=lang,
                            layout=layout,
                            auto_f_stack=not skipfs,
                            auto_quit=autoquit,
                            reposition_offset=offset,
                            lowest_z=lowest)
    controller.start()


@cli.command()
@click.argument(dcls=[None, '--port', '-p'],
                type=str,
                default='COM5',
                prompt='COM port of printer',
                help='COM port of the 3D printer')
@click.option(dcls=['--lang', '-l'],
              type=str,
              default="en",
              help='Language of the interface (en/fr)')
@click.option('--layout',
              type=str,
              default='QWERTY',
              help='Layout of the keyboard (QWERTY/AZERTY)')
@click.option(dcls=["--skipfs", "-s"],
              is_flag=True,
              flag_value=True,
              default=False,
              help='disable the automatic focus-stacking of pictures after a scan')
@click.option(dcls=['--autoquit', '-q'],
              is_flag=True,
              flag_value=True,
              default=False,
              help='sashimi quits automatically after scanning')
@click.option(dcls=['--offset', '-o'],
              type=int,
              default=1000,
              help='z offset in top-down mode')
@click.option(dcls=['--lowest', '-z'],
              is_flag=True,
              flag_value=True,
              default=False,
              help='simplifies z correction')
def multiple_exp(port, lang, layout, skipfs, autoquit, offset, lowest):
    user_path, exp_values = dialog_for_path_and_values()
    print("Input collection finished, the scanning program will start.")
    controller = Controller(user_path,
                            port,
                            lang=lang,
                            layout=layout,
                            multi_exp=exp_values,
                            auto_f_stack=not skipfs,
                            auto_quit=autoquit,
                            reposition_offset=offset,
                            lowest_z=lowest)
    controller.start()


@cli.command()
@click.option('--dir',
              type=str,
              prompt='Directory containing stacks',
              help='Directory containing subdirectories of image stacks')
def stack(dir):
    focus_stack.stack(dir)


@cli.command()
@click.option('--dir',
              type=str,
              prompt='Directory containing stacks',
              help='Directory containing subdirectories of image stacks')
def helicon_stack(dir):
    helicon_stacker.stack(dir)


if __name__ == "__main__":
    cli()
