#!/usr/bin/python

import argparse
import logging
import sys
import ast

import numpy

LOGGER = logging.getLogger()

def get_names(expr):
    tree = ast.parse(expr)
    return get_names_rec(tree)

def union(sets):
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
    elif isinstance(node, ast.Name):
        return set([node.id])
    elif isinstance(node, ast.Attribute):
        return  get_names_rec(node.value)
    elif isinstance(node, ast.Num):
        return set()
    elif isinstance(node, ast.Tuple):
        return set.union(*map(get_names_rec, node.elts))
    elif isinstance(node, ast.List):
        return set.union(*map(get_names_rec, node.elts))
    elif isinstance(node, ast.Slice):
        children = (node.lower, node.step, node.upper)
        return set.union(*map(get_names_rec, [x for x in children if x is not None]))
    else:
        raise ValueError(node)

def uses_stdin(expr):
    names = get_names(expr)
    return 'd' in names or 'data' in names

def main():
    parser = argparse.ArgumentParser(description='Interact with numpy from the command line')
    parser.add_argument('expr', type=str, help='Expression involving d, a numpy array')
    parser.add_argument('--debug', action='store_true', help='Print debug output')
    parser.add_argument('data_sources', type=str, nargs='*', help='Files to read data from. Stored in d1, d2 etc')

    parser.add_argument('--input-format', '-I', type=str, help='Dtype of the data read in')

    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('--raw-format', '-R', type=str, help='Output as a flat numpy array with this format')
    format_group.add_argument('--raw', action='store_true', help='Result is a string that should be written to standard out')
    format_group.add_argument('--repr', action='store_true', help='Output a repr of the result')
    parser.add_argument(
        '--module', '-m',
        action='append',
        help='Result is a string that should be written to standard out')

    args = parser.parse_args(sys.argv[1:])
    args.module = args.module or []

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    module_dict = dict()
    for m in args.module:
        module_dict.update(**imp(m))

    module_dict['np'] = numpy
    LOGGER.debug('Module dict: %r', module_dict)

    context = module_dict.copy()

    if uses_stdin(args.expr):
        data = read_data(args.input_format, sys.stdin)
        context.update(data=data, d=data)
    else:
        LOGGER.debug('takes no data')

    for index, source in enumerate(args.data_sources):
        with open(source) as stream:
            data = read_data(args.input_format, stream)
            name1 = 'd{}'.format(index + 1)
            name2 = 'data{}'.format(index + 1)
            context.update({name1: data, name2: data})

    LOGGER.debug('context: %r', module_dict)

    result = multiline_eval(args.expr, context)
    if isinstance(result, (float, int, numpy.number)):
        result = numpy.array([result])

    LOGGER.debug('Result length: %s ', len(result))

    if args.raw:
        sys.stdout.write(result)
    elif args.raw_format:
        sys.stdout.write(numpy.array(result, dtype=args.raw_format))
    elif args.repr:
        sys.stdout.write(repr(result))
    else:
        numpy.savetxt(sys.stdout, result, fmt='%s')

def multiline_eval(expr, context):
    "Evaluate several lines of input, returning the result of the last line"
    tree = ast.parse(expr)
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1])
    exec(compile(exec_expr, 'file', 'exec'), context) #pylint: disable=exec-used
    return eval(compile(eval_expr, 'file', 'eval'), context) #pylint: disable=eval-used

def read_data(input_format, stream):
    if input_format is not None:
        data = numpy.fromstring(stream.read(), dtype=input_format)
    else:
        data = numpy.array([map(float, line.split()) for line in stream.read().splitlines()])
        if data.shape[1] == 1:
            # Treat a stream of numbers a 1-D array
            data = data.flatten()

    LOGGER.debug('Data length: %s ', len(data))
    LOGGER.debug('Data shape: %s ', data.shape)
    LOGGER.debug('data: %r', data)
    return data

def imp(s):
    name = s.split('.')[0]
    return {name: __import__(s)}
