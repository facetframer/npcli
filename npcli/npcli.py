#!/usr/bin/python

# make code as python 3 compatible as possible
from __future__ import absolute_import, division, print_function, unicode_literals

import autopep8
import argparse
import logging
import sys
import ast
from io import BytesIO

import numpy

LOGGER = logging.getLogger()

def get_names(expr):
    tree = ast.parse(expr)
    return get_names_rec(tree)

def union(sets):
    sets = list(sets)
    return set.union(*sets) if sets else set()

def get_names_rec(node):
    if isinstance(node, ast.Call):
        arg_names = map(get_names, node.args)
        return get_names_rec(node.func) | union(arg_names)
    elif isinstance(node, ast.Module):
        return union(map(get_names_rec, node.body))
    elif isinstance(node, ast.Expr):
        return get_names_rec(node.value)
    elif isinstance(node, ast.Assign):
        return get_names_rec(node.value)
    elif isinstance(node, ast.Index):
        return get_names_rec(node.value)
    elif isinstance(node, ast.Subscript):
        return get_names_rec(node.value) | get_names_rec(node.slice)
    elif isinstance(node, ast.BinOp):
        return get_names_rec(node.left) | get_names_rec(node.right)
    elif isinstance(node, ast.Compare):
        return get_names_rec(node.left) | union(map(get_names_rec, node.comparators))
    elif isinstance(node, ast.UnaryOp):
        LOGGER.debug(dir(node.operand))
        return get_names_rec(node.operand)
    elif isinstance(node, ast.Str):
        return set()
    elif isinstance(node, ast.Name):
        return set([node.id])
    elif isinstance(node, ast.Attribute):
        return  get_names_rec(node.value)
    elif isinstance(node, ast.Num):
        return set()
    elif isinstance(node, ast.Tuple):
        return union(map(get_names_rec, node.elts))
    elif isinstance(node, ast.List):
        return union(map(get_names_rec, node.elts))
    elif isinstance(node, ast.comprehension):
        return (union(map(get_names_rec, node.ifs)) | get_names_rec(node.iter) | get_names_rec(node.target))
    elif isinstance(node, ast.ListComp):
        return get_names_rec(node.elt) | union(map(get_names_rec, node.generators))
    elif isinstance(node, ast.Slice):
        children = (node.lower, node.step, node.upper)
        true_children = [x for x in children if x is not None]
        return union(map(get_names_rec, true_children)) if true_children else set()
    elif isinstance(node, ast.ExtSlice):
        return union(map(get_names_rec, node.dims)) if node.dims else set()
    else:
        raise ValueError(node)

def uses_stdin(expr):
    names = get_names(expr)
    return 'd' in names or 'data' in names

def build_parser():
    parser = argparse.ArgumentParser(description='Interact with numpy from the command line')
    parser.add_argument('expr', type=str, help='Expression involving d, a numpy array')
    parser.add_argument(
        '--expr',
        '-e',
        type=str,
        help='Expression involving d, a numpy array. Multipe expressions get chained',
        dest='more_expressions',
        action='append',
        metavar='EXPR')

    parser.add_argument('--code', action='store_true', default=False, help='Produce python code rather than running')
    parser.add_argument('--debug', action='store_true', help='Print debug output')
    parser.add_argument('data_sources', type=str, nargs='*', help='Files to read data from. Stored in d1, d2 etc')
    parser.add_argument('--input-format', '-I', type=str, help='Dtype of the data read in. "lines" for a list of lines. "str" for a string. "csv" for csv, "pandas" for a pandas csv')
    parser.add_argument('--kitchen-sink', '-K', action='store_true', help='Import a lot of useful things into the execution scope')
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        '--output-format',
        '-O',
        type=str,
        help='Output as a flat numpy array with this format. "str" for a string')
    format_group.add_argument('--raw', action='store_true', help='Result is a string that should be written to standard out')
    format_group.add_argument('--repr', '-D', action='store_true', help='Output a repr of the result. Often used for _D_ebug')
    format_group.add_argument('--no-result', '-n', action='store_true', help="Discard result")
    parser.add_argument(
        '--module', '-m',
        action='append',
        help='Result is a string that should be written to standard out')
    return parser


def run(stdin_stream, args):
    parser = build_parser()
    args = parser.parse_args(args)
    args.module = args.module or []

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    module_dict = dict()
    for m in args.module:
        module_dict.update(**imp(m))

    if args.kitchen_sink:
        # Lazy because these may not be installed
        import pandas
        import pylab
        import pandas

        for x in [pandas, numpy, pylab]:
            module_dict.update(imp_all(x))

        module_dict['numpy'] = numpy
        module_dict['pylab'] = pylab
        module_dict['pandas'] = pandas
        module_dict['pd'] = pandas

    module_dict['np'] = numpy
    module_dict['numpy'] = numpy
    LOGGER.debug('Module dict: %r', module_dict)

    context = module_dict.copy()
    expressions = ([args.expr] if args.expr else []) + (args.more_expressions or [])

    if not expressions:
        return

    if uses_stdin(expressions[0]):
        data = read_data(args.input_format, stdin_stream)
        context.update(data=data, d=data)
    else:
        LOGGER.debug('takes no data')


    if args.code:
        # Lazy import because this is big
        import autopep8
        program = '\n'.join(expressions) + '\n'
        return autopep8.fix_string(program)

    for index, source in enumerate(args.data_sources):
        with open(source) as stream:
            data = read_data(args.input_format, stream)
            name1 = 'd{}'.format(index + 1)
            name2 = 'data{}'.format(index + 1)
            context.update({name1: data, name2: data})

    LOGGER.debug('context: %r', module_dict)

    for expr in expressions:
        context['d'] = multiline_eval(expr, context)

    result = context['d']

    if isinstance(result, (float, int, numpy.number)):
        result = numpy.array([result])

    try:
        LOGGER.debug('Result length: %f ', len(result))
    except TypeError:
        LOGGER.debug('Result has no length length: %r! ', result)

    if args.no_result:
        return tuple()
    elif args.raw:
        return (result,)
    elif args.output_format:
        if args.output_format == 'str':
            return result
        else:
            return (numpy.array(result, dtype=args.output_format),)
    elif args.repr:
        return (repr(result).encode('utf8'),)
    else:
        output = BytesIO()
        numpy.savetxt(output, result, fmt=b'%s')
        return (output.getvalue(),)

def main():
    for part in run(sys.stdin, sys.argv[1:]):
        sys.stdout.write(part)
        sys.stdout.flush()

def multiline_eval(expr, context):
    "Evaluate several lines of input, returning the result of the last line"
    tree = ast.parse(expr)
    is_eval = isinstance(tree.body[-1], ast.Expr)

    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1])

    if is_eval:
        final_eval_expr = ast.Expression(tree.body[-1].value)
    else:
        final_exec_expr = ast.Module([tree.body[-1]])

    exec(compile(exec_expr, 'file', 'exec'), context) #pylint: disable=exec-used

    if is_eval:
        return eval(compile(eval_expr, 'file', 'eval'), context) #pylint: disable=eval-used
    else:
        exec(compile(final_exec_expr, 'file', 'exec'), context) #pylint: disable=exec-used
        return None

def maybe_float(x):
    try:
        return float(x)
    except ValueError:
        return x


def read_data(input_format, stream):
    if input_format == 'lines':
        data = [x.decode('utf8').strip('\n') for x in stream.readlines()]
    elif input_format == 'str':
        return stream.read()
    elif input_format == 'csv':
        data = numpy.genfromtxt(stream, delimiter=',')
    elif input_format == 'pandas':
        import pandas
        data = pandas.DataFrame.from_csv(stream)
    elif input_format is not None:
        data = numpy.fromstring(stream.read(), dtype=input_format)

    else:
        data = numpy.array([list(map(maybe_float, line.split())) for line in stream.read().splitlines()])
        if len(data.shape) > 1 and data.shape[1] == 1:
            # Treat a stream of numbers a 1-D array
            data = data.flatten()

    LOGGER.debug('Data length: %s ', len(data))

    if hasattr(data, 'shape'):
        LOGGER.debug('Data shape: %s ', data.shape)
    else:
        LOGGER.debug('Data shape: None')

    LOGGER.debug('data: %r', data)
    return data

def imp(s):
    name = s.split('.')[0]
    return {name: __import__(s)}

def imp_all(s):
    if isinstance(s, str):
        name = s.split('.')[0]
        obj = __import__(s)
        for x in name[1:]:
            obj = getattr(obj, x)
    else:
        obj = s

    if hasattr(obj, '__all__'):
        return dict((k, getattr(obj, k)) for k in obj.__all__)
    else:
        return dict(vars(obj))
