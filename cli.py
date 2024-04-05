# -*- coding:utf-8 -*-

import os
import argparse
from utils.store import generate_vector_store

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='书生浦语农业助手')
    parser.add_argument('--cmd', type=str,
                        help='要执行的命令，db 更新向量数据库')
    parser.add_argument('--doc_path',type=str,
                        help='文档路径')
    parser.add_argument('--db_path',  type=str,
                        help='向量数据库路径')
    parser.add_argument('--embedding_model', type=str,
                        help='词向量模型名')
    return parser.parse_args()

if __name__ == '__main__':
    arg_dict = parse_args()

    if 'db' == arg_dict.cmd:
        arg_check = True
        if arg_dict.doc_path is None or arg_dict.doc_path == '' or not os.path.exists(arg_dict.doc_path):
            arg_check = False
            print('--doc_path 文档路径参数不能为空或不是有效的路径')
        if arg_dict.db_path is None or arg_dict.db_path == '' or not os.path.exists(arg_dict.db_path):
            arg_check = False
            print('--db_path 向量数据库路径参数不能为空或不是有效的路径')
        if arg_dict.embedding_model is None or arg_dict.embedding_model == '' or not os.path.exists(arg_dict.embedding_model):
            arg_check = False
            print('--embedding_model 词向量模型路径参数不能为空或不是有效的路径')
        if arg_check:
            generate_vector_store(arg_dict.doc_path, arg_dict.db_path, arg_dict.embedding_model)
    else:
        print('参数错误')

