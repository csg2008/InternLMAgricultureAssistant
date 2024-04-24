# -*- coding:utf-8 -*-

import os
import argparse

from utils.store import generate_vector_store
from action.knowledge import KnowledgeQuery

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='书生浦语农业助手')
    parser.add_argument('--cmd', type=str,
                        help='要执行的命令, db 更新向量数据库, query 从向量数据库查询问题')
    parser.add_argument('--doc_path',type=str,
                        help='文档路径')
    parser.add_argument('--db_path',  type=str,
                        help='向量数据库路径')
    parser.add_argument('--embedding_model', type=str,
                        help='词向量模型名')
    return parser.parse_args()

def check_db_param(args):
    status = True
    msg = ''

    if args.doc_path is None or not os.path.exists(args.doc_path):
        status = False
        msg = '--doc_path 文档路径参数不能为空或不是有效的路径'
    if args.db_path is None or not os.path.exists(args.db_path):
        status = False
        msg = '--db_path 向量数据库路径参数不能为空或不是有效的路径'
    if args.embedding_model is None or not os.path.exists(args.embedding_model):
        status = False
        msg = '--embedding_model 词向量模型路径参数不能为空或不是有效的路径'

    return status, msg

if __name__ == '__main__':
    arg_dict = parse_args()

    if 'db' == arg_dict.cmd:
        arg_check, check_msg = check_db_param(arg_dict)
        if arg_check:
            generate_vector_store(arg_dict.doc_path, arg_dict.db_path, arg_dict.embedding_model)
        else:
            print(check_msg)
    elif 'query' == arg_dict.cmd:
        arg_check, check_msg = check_db_param(arg_dict)
        if arg_check:
            question = '菠萝是长在树上的吗'
            retriever = KnowledgeQuery(arg_dict.db_path, arg_dict.embedding_model)
            result = retriever.run(question)
            print(result.result[0]['content'])
        else:
            print(check_msg)
    else:
        print('参数错误')

