from flask import Flask, render_template, request, url_for, redirect,session
from matplotlib.pyplot import table
from data_load import load_wikisql_data
# from question_server import my_infer
import os,argparse
# from sqlnet.dbengine import DBEngine
# from sqlova.utils.utils_wikisql import *
# from train import construct_hyper_param, get_models
# import stanza.server as corenlp

app = Flask(__name__)
app.secret_key = 'Naveen'

path_h = os.getcwd()
data_path = os.path.join(path_h, "data")
split = "dev"

## loading data
dev_data, dev_table = load_wikisql_data(data_path, mode=split, toy_model=False, toy_size=12, no_hs_tok=True)
list_dev_table = [(k,v) for k,v in dev_table.items()]
subset_dev_table = { k:v for k,v in list_dev_table[1:6] }

d = {1:'ABC',2:'DEF',3:'GHI',4:'JKL',5:'MNO'}
j = 1
for i in subset_dev_table.values():
    print(i)
    i['table_name'] = d[j]
    j+=1

# ## loading model
# ## Set up hyper parameters and paths
# parser = argparse.ArgumentParser()
# parser.add_argument("--path_h", default=os.getcwd(), type=str, help='model file to use (e.g. model_best.pt)')
# args = construct_hyper_param(parser)

# # path_h = args.path_h or os.getcwd()

# args.model_file = os.path.join(path_h, 'pretrained', 'model_best.pt')
# args.bert_model_file = os.path.join(path_h, 'pretrained', 'model_bert_best.pt')
# args.bert_path = os.path.join(path_h, 'data')
# args.data_path = os.path.join(path_h, 'data')
# args.split = 'dev'

# BERT_PT_PATH = args.bert_path

# # Load pre-trained models
# path_model_bert = args.bert_model_file
# path_model = args.model_file
# args.no_pretraining = True  # counterintuitive, but avoids loading unused models
# model, model_bert, tokenizer, bert_config = get_models(args, BERT_PT_PATH, trained=True, path_model_bert=path_model_bert, path_model=path_model)

# client = corenlp.CoreNLPClient(annotators='tokenize,ssplit'.split(','), endpoint='http://localhost:9001')

@app.route('/')
def home():
    return render_template('home.html',table_data=subset_dev_table)

@app.route('/query',methods=["POST","GET"])
def query():

    if request.method == "GET":
        return redirect("/")

    nlu1 = request.form.get('question')
    table_id = request.form.get('tableid')
    # pr_sql_i, pr_ans = my_infer(nlu1, table_id, dev_table, args.data_path, 'dev', model, model_bert, bert_config, args.max_seq_length, args.num_target_layers, beam_size=4, show_table=False, show_answer_only=False)

    sqlquery = nlu1
    result = table_id

    session['message'] = ""

    if result == '':
        session['message'] = 'Enter Valid Question!'
        return redirect('/')

    return render_template('results.html',question=nlu1,sqlquery=sqlquery,result=result)

if __name__ == "__main__":
    app.run(debug=True)