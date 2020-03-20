import click

from prlab.gutils import command_run, run_k_fold

"""
General run by call package with configure from file (JSON) or command line args
Now support two sub-command: run and k_fold
Usage:
    python -m prlab.cli run --json_conf2 tmp/t.json --run to-del
    or
    python -m prlab.cli k_fold --call prlab.medical.k_fold.run_one_fold --json_conf config/medicine.json
"""


@click.group()
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    else:
        click.echo('I am about to invoke %s' % ctx.invoked_subcommand)


cli.add_command(command_run, name='run')
cli.add_command(run_k_fold, name='k_fold')

if __name__ == '__main__':
    cli()
