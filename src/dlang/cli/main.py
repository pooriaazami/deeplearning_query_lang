# Main entry point for the dlang command-line tool
import argparse
import sys
from .commands import init, parse, run, validate, plugin  # we'll create these modules

def main():
    parser = argparse.ArgumentParser(description='Deep Learning Description Language Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Create a new project')
    init_parser.add_argument('project_name', help='Name of the project to create')
    init_parser.add_argument('--directory', '-d', default='.', help='Directory to create the project in')

    # parse command
    parse_parser = subparsers.add_parser('parse', help='Parse a .dl file and generate code')
    parse_parser.add_argument('input_file', help='Path to the .dl file')
    parse_parser.add_argument('--output', '-o', help='Output directory for generated code')
    parse_parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'jax'], default='pytorch', help='Target backend')

    # run command
    run_parser = subparsers.add_parser('run', help='Parse .dl file, generate code, and run it')
    run_parser.add_argument('input_file', help='Path to the .dl file')
    run_parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'jax'], default='pytorch', help='Target backend')
    run_parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help='Device to run on')

    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a .dl file for syntax and semantics')
    validate_parser.add_argument('input_file', help='Path to the .dl file')

    # plugin command
    plugin_parser = subparsers.add_parser('plugin', help='Manage plugins')
    plugin_subparsers = plugin_parser.add_subparsers(dest='plugin_command', help='Plugin commands')

    plugin_list = plugin_subparsers.add_parser('list', help='List available plugins')
    plugin_install = plugin_subparsers.add_parser('install', help='Install a plugin')
    plugin_install.add_argument('plugin_name', help='Name of the plugin to install')
    plugin_install.add_argument('--version', help='Specific version to install')
    plugin_uninstall = plugin_subparsers.add_parser('uninstall', help='Uninstall a plugin')
    plugin_uninstall.add_argument('plugin_name', help='Name of the plugin to uninstall')

    # format command
    format_parser = subparsers.add_parser('format', help='Format a .dl file')
    format_parser.add_argument('input_file', help='Path to the .dl file to format')
    format_parser.add_argument('--output', '-o', help='Output file (defaults to overwriting input)')

    # doc command
    doc_parser = subparsers.add_parser('doc', help='Generate documentation from a .dl file')
    doc_parser.add_argument('input_file', help='Path to the .dl file')
    doc_parser.add_argument('--output', '-o', help='Output directory for generated documentation')

    # experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run training/evaluation experiments')
    exp_parser.add_argument('input_file', help='Path to the .dl file')
    exp_parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'jax'], default='pytorch', help='Target backend')
    exp_parser.add_argument('--tracking', choices=['wandb', 'mlflow', 'none'], default='none', help='Experiment tracking backend')

    # test command
    test_parser = subparsers.add_parser('test', help='Test generated code or model')
    test_parser.add_argument('input_file', help='Path to the .dl file or generated code')
    test_parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'jax'], default='pytorch', help='Target backend')

    # deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Export model for deployment')
    deploy_parser.add_argument('input_file', help='Path to the .dl file or generated model')
    deploy_parser.add_argument('--format', choices=['onnx', 'torchscript', 'saved_model', 'tflite'], default='onnx', help='Export format')
    deploy_parser.add_argument('--output', '-o', help='Output directory for exported model')

    # backend command
    backend_parser = subparsers.add_parser('backend', help='Manage backends')
    backend_subparsers = backend_parser.add_subparsers(dest='backend_command', help='Backend commands')
    backend_list = backend_subparsers.add_parser('list', help='List available backends')
    backend_set = backend_subparsers.add_parser('set', help='Set default backend')
    backend_set.add_argument('backend', choices=['pytorch', 'tensorflow', 'jax'], help='Backend to set as default')

    # version command
    version_parser = subparsers.add_parser('version', help='Show version information')

    # config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config commands')
    config_get = config_subparsers.add_parser('get', help='Get a configuration value')
    config_get.add_argument('key', help='Configuration key')
    config_set = config_subparsers.add_parser('set', help='Set a configuration value')
    config_set.add_argument('key', help='Configuration key')
    config_set.add_argument('value', help='Configuration value')

    args = parser.parse_args()

    if args.command == 'init':
        init.init_project(args)
    elif args.command == 'parse':
        parse.parse_file(args)
    elif args.command == 'run':
        run.run_file(args)
    elif args.command == 'validate':
        validate.validate_file(args)
    elif args.command == 'plugin':
        if args.plugin_command == 'list':
            plugin.list_plugins(args)
        elif args.plugin_command == 'install':
            plugin.install_plugin(args)
        elif args.plugin_command == 'uninstall':
            plugin.uninstall_plugin(args)
        else:
            plugin_parser.print_help()
    elif args.command == 'format':
        # TODO: implement format
        print(f"Formatting {args.input_file}")
    elif args.command == 'doc':
        # TODO: implement doc
        print(f"Generating documentation for {args.input_file}")
    elif args.command == 'experiment':
        # TODO: implement experiment
        print(f"Running experiment for {args.input_file}")
    elif args.command == 'test':
        # TODO: implement test
        print(f"Testing {args.input_file}")
    elif args.command == 'deploy':
        # TODO: implement deploy
        print(f"Deploying {args.input_file}")
    elif args.command == 'backend':
        if args.backend_command == 'list':
            print("Available backends: pytorch, tensorflow, jax")
        elif args.backend_command == 'set':
            print(f"Setting default backend to {args.backend}")
        else:
            backend_parser.print_help()
    elif args.command == 'version':
        from . import __version__
        print(f"dlang version {__version__}")
    elif args.command == 'config':
        if args.config_command == 'get':
            print(f"Getting config for {args.key}")
        elif args.config_command == 'set':
            print(f"Setting config {args.key} = {args.value}")
        else:
            config_parser.print_help()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()