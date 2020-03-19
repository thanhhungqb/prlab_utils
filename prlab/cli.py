import click

from prlab.gutils import command_run, run_k_fold

"""
General run by call package with configure from file (JSON) or command line args
Usage:
    python -m prlab.run --run_id test --call prlab.emotion.ferplus.sr_stn_vgg_8classes.train_test_control \
        --run 8-sr-stn-adam+sgd-05
"""


@click.group(invoke_without_command=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def cli(ctx, debug):
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    else:
        click.echo('I am about to invoke %s' % ctx.invoked_subcommand)


cli.add_command(command_run, name='run')
cli.add_command(run_k_fold, name='k_fold')

if __name__ == '__main__':
    cli()
