from clang.cindex import Config
import argparse
import time
from pathlib import Path
import multiprocessing
import os
from tqdm import tqdm
from parser_process import ParserProcess

file_types = ('*.c', '*.cc', '*.cpp', '*.cxx', '*.c++')


def files(input_path):
    if os.path.isfile(input_path):
        yield input_path
    for file_type in file_types:
        for file_path in Path(input_path).rglob(file_type):
            yield file_path.as_posix()


def main():
    args_parser = argparse.ArgumentParser(
        description='cppminer generates a code2seq dataset from C++ sources')

    args_parser.add_argument('Path',
                             metavar='path',
                             type=str,
                             help='the path sources directory')

    args_parser.add_argument('OutPath',
                             metavar='out',
                             type=str,
                             help='the output path')

    args_parser.add_argument('-c', '--max_contexts_num',
                             metavar='contexts-number',
                             type=int,
                             help='maximum number of contexts per sample',
                             default=100,
                             required=False)

    args_parser.add_argument('-l', '--max_path_len',
                             metavar='path-length',
                             type=int,
                             help='maximum path length (0 - no limit)',
                             default=0,
                             required=False)

    args_parser.add_argument('-s', '--max_subtokens_num',
                             metavar='subtokens-num',
                             type=int,
                             help='maximum number of sub-tokens in a token (0 - no limit)',
                             default=5,
                             required=False)

    args_parser.add_argument('-d', '--max_ast_depth',
                             metavar='ast-depth',
                             type=int,
                             help='maximum depth of AST (0 - no limit)',
                             default=0,
                             required=False)

    args_parser.add_argument('-p', '--processes_num',
                             metavar='processes-number',
                             type=int,
                             help='number of parallel processes',
                             default=4,
                             required=False)

    args_parser.add_argument('-e', '--libclang',
                             metavar='libclang-path',
                             type=str,
                             help='path to libclang.so file',
                             required=False)

    args_parser.add_argument('-w', '--window',
                             metavar='window',
                             type=int,
                             help='window of code snippet (0 - entire function)',
                             default=0,
                             required=False)

    args_parser.add_argument('-ws', '--window_step',
                             metavar='window-step',
                             type=int,
                             help='the step of window for code snippets (0 - same as window)',
                             default=0,
                             required=False)

    args = args_parser.parse_args()

    if args.libclang:
        # File path example '/usr/lib/llvm-6.0/lib/libclang.so'
        Config.set_library_file(args.libclang)

    parallel_processes_num = args.processes_num
    print('Parallel processes num: ' + str(parallel_processes_num))

    max_contexts_num = args.max_contexts_num
    print('Max contexts num: ' + str(max_contexts_num))

    max_path_len = args.max_path_len
    print('Max path length: ' + str(max_path_len))

    max_subtokens_num = args.max_subtokens_num
    print('Max sub-tokens num: ' + str(max_subtokens_num))

    max_ast_depth = args.max_ast_depth
    print('Max AST depth: ' + str(max_ast_depth))

    input_path = Path(args.Path).resolve().as_posix()
    print('Input path: ' + input_path)

    output_path = Path(args.OutPath).resolve().as_posix()
    print('Output path: ' + output_path)

    window = args.window
    print('Window: ' + str(window))

    window_step = args.window_step
    print('Window step: ' + str(window_step))

    print("Parsing files ...")
    tasks = multiprocessing.JoinableQueue()
    if parallel_processes_num == 1:
        parser = ParserProcess(tasks, max_contexts_num, max_path_len, max_subtokens_num, max_ast_depth, input_path,
                               output_path, window, window_step)
        for file_path in files(input_path):
            print("Parsing : " + file_path)
            tasks.put(file_path)
            parser.parse_file()
        parser.save()
        tasks.join()
    else:
        processes = [ParserProcess(tasks, max_contexts_num, max_path_len, max_subtokens_num, max_ast_depth, input_path,
                                   output_path, window, window_step)
                     for _ in range(parallel_processes_num)]
        for p in processes:
            p.start()

        for file_path in files(input_path):
            tasks.put(file_path)

        # add terminating tasks
        for i in range(parallel_processes_num):
            tasks.put(None)

        # Wait for all of the tasks to finish
        tasks_left = tasks.qsize()
        with tqdm(total=tasks_left) as pbar:
            while tasks_left > 0:
                time.sleep(1)
                tasks_num = tasks.qsize()
                pbar.update(tasks_left - tasks_num)
                tasks_left = tasks_num

        tasks.join()
        for p in processes:
            p.join()
    print("Parsing done")


if __name__ == '__main__':
    main()
