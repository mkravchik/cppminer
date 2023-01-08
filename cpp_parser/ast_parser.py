from clang.cindex import Index
from .sample import Sample
from .context import Context
from .path import Path
from .ast_utils import ast_to_graph, is_function, is_class, is_operator_token, is_namespace, make_ast_err_message
from networkx.algorithms import shortest_path
from networkx.drawing.nx_agraph import to_agraph
from itertools import combinations
import uuid
import os
import re
import random
from ClassMap.classMap import mapper


def debug_save_graph(func_node, g):
    file_name = func_node.spelling + ".png"
    num = 0
    while os.path.exists(file_name):
        file_name = func_node.spelling + str(num) + ".png"
        num += 1
    a = to_agraph(g)
    a.draw(file_name, prog='dot')
    a.clear()


def tokenize(name, max_subtokens_num):
    if is_operator_token(name):
        return [name]
    first_tokens = name.split('_')
    str_tokens = []
    for token in first_tokens:
        internal_tokens = re.findall('[a-z]+|[A-Z]+[a-z]*|[0-9.]+|[-*/&|%=()]+', token)
        str_tokens += [t for t in internal_tokens if len(t) > 0]
    assert len(str_tokens) > 0, "Can't tokenize expr: {0}".format(name)
    if max_subtokens_num != 0:
        str_tokens = str_tokens[:max_subtokens_num]
    return str_tokens


class AstParser:
    def __init__(self, max_contexts_num, max_path_len, max_subtokens_num, max_ast_depth, out_path, input_path, window=0, step=0):
        self.validate = False
        self.save_buffer_size = 1000
        self.out_path = out_path
        self.max_subtokens_num = max_subtokens_num
        self.max_contexts_num = max_contexts_num
        self.max_path_len = max_path_len
        self.max_ast_depth = max_ast_depth
        self.index = Index.create()
        self.samples = set()
        self.header_only_functions = set()
        self.class_mapper = mapper("./ClassMap/classMap.json", input_path)
        self.file_class = {}
        self.window = window
        self.step = step

    def __del__(self):
        self.save()

    def __parse_node(self, node):
        try:
            namespaces = [x for x in node.get_children() if is_namespace(x)]
            for n in namespaces:
                # ignore standard library functions
                if n.displayname != 'std' and not n.displayname.startswith('__'):
                    self.__parse_node(n)

            functions = [x for x in node.get_children() if is_function(x)]
            for f in functions:
                self.__parse_function(f)

            classes = [x for x in node.get_children() if is_class(x)]
            for c in classes:
                methods = [x for x in c.get_children() if is_function(x)]
                for m in methods:
                    self.__parse_function(m)
        except Exception as e:
            if 'Unknown template argument kind' not in str(e):
                msg = make_ast_err_message(str(e), node)
                raise Exception(msg)

        self.__dump_samples()

    def get_ifdefs(self, file_path: str) -> list:
        """This function extracts the ifdefs from the file
        Clang does not have any ifdefs defined thus it skips everything under ifdef
        Alternatively, we could remove the ifdefs, but for that we need to use the parser 
        that will find the matching #else/#endif and I did not find a way to do it yet.
        Note: this is a very primitive mechanism as it does not work recursively on the file's includes
        Args:
            file_path (_type_): _description_
        """    
        # index = clang.cindex.Index.create()
        # tu = index.parse(file_path, args=['-x', 'c++'])
        # for x in tu.cursor.get_tokens():
        #     print (x.kind)
        #     print ("  " + srcrangestr(x.extent))
        #     print ("  '" + str(x.spelling) + "'")
        res = []
        try:
            with open(file_path) as src:
                for line in src.readlines(): 
                    # taking care of the simple cases
                    # single line, no conditions
                    # A proper treatment requires using a proper preprocessor
                    match = re.findall("#if defined\(([A-Za-z0-9_-]+)\)$", line)
                    if len(match) == 0:
                        match = re.findall("#ifdef\(([A-Za-z0-9_-]+)\)$", line)
                    if len(match) == 0:
                        match = re.findall("#ifdef\s+([A-Za-z0-9_-]+)\s*$", line)
                    if len(match):
                        res.append(match[0])
        except Exception as e:
            print(f"Failed parsing {file_path}, error {e}")
            return res

        return res

    def parse(self, compiler_args, file_path=None):
        file_path_ = file_path if file_path is not None else compiler_args[0]
        if not file_path_ in self.file_class:
            label, inc_dirs, project, defines, target_set = self.class_mapper.getFileClass(file_path_)
            self.file_class[file_path_] = (label, inc_dirs, defines, target_set)

        for define in defines:
            compiler_args.extend(["-D", define])
        expected_defines = set(self.get_ifdefs(file_path_))
        for define in expected_defines:
            compiler_args.extend(["-D", define])

        # print(args)

        if inc_dirs is not None:
            for inc in inc_dirs:
                compiler_args.extend(["-I", inc])

        try:
            ast = self.index.parse(file_path, compiler_args)
        except Exception as e:
            print(f"Failed parsing {file_path}, error {e}")
            return

        self.__parse_node(ast.cursor)

    def __dump_samples(self):
        if len(self.samples) >= self.save_buffer_size:
            self.save()

    def save(self):
        if not self.out_path:
            return
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if len(self.samples) > 0:
            file_name = os.path.join(self.out_path, str(uuid.uuid4().hex) + ".c2s")
            # print(file_name)
            with open(file_name, "w") as file:
                for sample in self.samples:
                    file.write(str(sample.source_mark) + str(sample) + "\n")
            self.samples.clear()

    def __parse_function(self, func_node):
        try:
            # ignore standard library functions
            if func_node.displayname.startswith('__'):
                return

            # detect header only function duplicates
            file_name = func_node.location.file.name
            source_mark = (file_name, func_node.extent.start.line, self.file_class[file_name][-1])
            if file_name.endswith('.h') and func_node.is_definition:
                # print('Header only function: {0}'.format(func_node.displayname))
                if source_mark in self.header_only_functions:
                    # print('Duplicate')
                    return
                else:
                    self.header_only_functions.add(source_mark)

            # key = tokenize(func_node.spelling, self.max_subtokens_num)
            key = self.file_class[file_name][0]
            if key == "Unknown":
                return # don't waste time

            g = ast_to_graph(func_node, self.max_ast_depth)

            terminal_nodes = [node for (node, degree) in g.degree() if degree == 1]
            # all_nodes = [node for node in g.nodes()]
            used_nodes = terminal_nodes # all_nodes
            random.shuffle(used_nodes)

            # if (file_name.find("ccm.c") > 0):
            #     debug_save_graph(func_node, g)
            #     print(file_name, func_node.displayname)
            #     nodes = []
            #     for (node, degree) in g.degree():
            #         if degree == 1:
            #             nodes.append((g.nodes[node]['label'], g.nodes[node]['loc']))
            #     sorted_nodes = sorted(nodes, key=lambda a: a[1])
            #     print(sorted_nodes)

            # if a window is specified, we need to create number of samples for this function
            windows = []
            if self.window:
                step = self.step if self.step != 0 else self.window
                w_start = func_node.extent.start.line
                while w_start < func_node.extent.end.line:
                    windows.append((w_start, w_start + self.window))
                    w_start += step
            else:
                windows.append((func_node.extent.start.line, func_node.extent.end.line))

            for window in windows:
                # print(f"processing window {window}")
                contexts = set()
                source_mark = (file_name, window[0], self.file_class[file_name][-1])
                ends = combinations(used_nodes, 2) # we want to create the full set each time
                for start, end in ends:
                    # print(f"start {g.nodes[start]['label']} {g.nodes[start]['loc']}, end {g.nodes[end]['label']} {g.nodes[end]['loc']},")
                    if g.nodes[start]['loc'] >= window[0] and g.nodes[start]['loc'] < window[1] and \
                       g.nodes[end]['loc'] >= window[0] and g.nodes[end]['loc'] < window[1]:
                        path = shortest_path(g, start, end)
                        if path:
                            # print(f"path {path}")
                            if self.max_path_len != 0 and len(path) > self.max_path_len:
                                continue  # skip too long paths
                            path = path[1:-1]
                            start_node = g.nodes[start]['label']
                            tokenize_start_node = not g.nodes[start]['is_reserved']
                            end_node = g.nodes[end]['label']
                            tokenize_end_node = not g.nodes[end]['is_reserved']

                            path_tokens = []
                            for path_item in path:
                                path_node = g.nodes[path_item]['label']
                                path_tokens.append(path_node)

                            context = Context(
                                tokenize(start_node, self.max_subtokens_num) if tokenize_start_node else [start_node],
                                tokenize(end_node, self.max_subtokens_num) if tokenize_end_node else [end_node],
                                Path(path_tokens, self.validate), self.validate)
                            contexts.add(context)
                        else:
                            # print("no path")
                            pass
                        if len(contexts) > self.max_contexts_num:
                            break

                if len(contexts) > 0:
                    sample = Sample(key, contexts, source_mark, self.validate)
                    self.samples.add(sample)
                    # print(f"Added {len(contexts)} contexts for window {window} for source mark {source_mark} {func_node.displayname, func_node.extent.start.line, func_node.extent.end.line}")
                else:
                    # print(f"No contexts for window {window} for source mark {source_mark} {func_node.displayname, func_node.extent.start.line, func_node.extent.end.line}")
                    pass

        except Exception as e:
            # skip unknown cursor exceptions
            if 'Unknown template argument kind' not in str(e):
                # print('Failed to parse function : ')
                # print('Filename : ' + func_node.location.file.name)
                # print('Start {0}:{1}'.format(func_node.extent.start.line, func_node.extent.start.column))
                # print('End {0}:{1}'.format(func_node.extent.end.line, func_node.extent.end.column))
                # print(e)
                pass
