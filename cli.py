# -*- coding:utf-8 -*-

import os
import argparse

from utils.retriever import Retriever
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
                        help='embedding 模型名')
    parser.add_argument('--reranker_model', type=str,
                        help='reranker 模型名')
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
    if args.reranker_model is None or not os.path.exists(args.reranker_model):
        status = False
        msg = '--reranker_model 词向量模型路径参数不能为空或不是有效的路径'

    return status, msg

if __name__ == '__main__':
    arg_dict = parse_args()

    if 'db' == arg_dict.cmd:
        arg_check, check_msg = check_db_param(arg_dict)
        if arg_check:
            retriever = Retriever(arg_dict.embedding_model, arg_dict.reranker_model)
            retriever.build(arg_dict.doc_path, arg_dict.db_path)
        else:
            print(check_msg)
    elif 'query' == arg_dict.cmd:
        arg_check, check_msg = check_db_param(arg_dict)
        if arg_check:
            question = '菠萝是长在树上的吗'
            retriever = Retriever(arg_dict.embedding_model, arg_dict.reranker_model, arg_dict.db_path)
            result = retriever.query(question)
            print(result)
        else:
            print(check_msg)
    else:
        print('参数错误')

