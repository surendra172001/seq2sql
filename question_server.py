import argparse, os
from sqlnet.dbengine import DBEngine
from sqlova.utils.utils_wikisql import *
from train import construct_hyper_param, get_models
import stanza.server as corenlp


def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok


def my_infer(nlu1,
          table_id, tables, path_db, db_name,
          model, model_bert, bert_config, max_seq_length, num_target_layers,
          beam_size=4, show_table=False, show_answer_only=False):
    # I know it is of against the DRY principle but to minimize the risk of introducing bug w, the infer function introuced.
    model.eval()
    model_bert.eval()
    engine = DBEngine(os.path.join(path_db, f"{db_name}.db"))

    # Get inputs
    nlu = [nlu1]
    nlu_t1 = tokenize_corenlp_direct_version(client, nlu1)
    nlu_t = [nlu_t1]

    tb1 = tables[table_id]
    hds1 = tb1['header']
    tb = [tb1]
    hds = [hds1]
    hs_t = [[]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_bert(bert_config, model_bert, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, engine, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size)

    # sort and generate
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, [tb1])
    pr_sql_q = [pr_sql_q1]

    try:
        pr_ans, _ = engine.execute_return_query(tb[0]['id'], pr_sc[0], pr_sa[0], pr_sql_i[0]['conds'])
    except:
        pr_ans = ['Answer not found.']
        pr_sql_q = ['Answer not found.']

    # if show_answer_only:
    #     print(f'Q: {nlu[0]}')
    #     print(f'A: {pr_ans[0]}')
    #     print(f'SQL: {pr_sql_q}')

    # else:
    #     print(f'START ============================================================= ')
    #     print(f'{hds}')
    #     print(f'nlu: {nlu}')
    #     print(f'pr_sql_i : {pr_sql_i}')
    #     print(f'pr_sql_q : {pr_sql_q}')
    #     print(f'pr_ans: {pr_ans}')
    #     print(f'---------------------------------------------------------------------')

    return pr_sql_i, pr_ans



## Set up hyper parameters and paths
parser = argparse.ArgumentParser()
parser.add_argument("--path_h", default=os.getcwd(), type=str, help='model file to use (e.g. model_best.pt)')
args = construct_hyper_param(parser)

path_h = args.path_h or os.getcwd()

args.model_file = os.path.join(path_h, 'pretrained', 'model_best.pt')
args.bert_model_file = os.path.join(path_h, 'pretrained', 'model_bert_best.pt')
args.bert_path = os.path.join(path_h, 'data')
args.data_path = os.path.join(path_h, 'data')
args.split = 'dev'

BERT_PT_PATH = args.bert_path

# Load pre-trained models
path_model_bert = args.bert_model_file
path_model = args.model_file
args.no_pretraining = True  # counterintuitive, but avoids loading unused models
model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)


_, tables = load_wikisql_data(args.data_path, mode=args.split, toy_model=args.toy_model, toy_size=args.toy_size, no_hs_tok=True)

print('Total tables: ', len(tables))

client = corenlp.CoreNLPClient(annotators='tokenize,ssplit'.split(','), endpoint='http://localhost:9001')

nlu1 = "Which company have more than 100 employees?"
table_id = ''
path_db = args.data_path

n_Q = 100000
for i in range(n_Q):
    if n_Q > 1:
        nlu1 = input('Type question: ')
        table_id = input('Enter table id: ')
    pr_sql_i, pr_ans = my_infer(nlu1, table_id, tables, path_db, 'dev', model, model_bert, bert_config, args.max_seq_length, args.num_target_layers, beam_size=4, show_table=False, show_answer_only=False)


