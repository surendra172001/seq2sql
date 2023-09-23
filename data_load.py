import os
import json

def load_wikisql_data(path_wikisql, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    if aug:
        mode = f"aug.{mode}"
        print('Augmented data is loaded!')

    path_sql = os.path.join(path_wikisql, mode+'_tok.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table

# path_h = os.getcwd()
# data_path = os.path.join(path_h, "data")
# split = "dev"
# dev_data, dev_table = load_wikisql_data(data_path, mode=split, toy_model=False, toy_size=12, no_hs_tok=True)

# print(dev_table.keys())


'''
// text += '<table class="table"><thead><tr><th scope="col">Columns</th><th scope="col">Possible Values</th></tr><tbody>';

      //old
      headers.forEach(e => {
        text += ('<th scope="col">'+e+'</th>');
      });
      text += ('</tr></thead></table>');

      //new
      // rowValues = jsonData[table_object.value]['rows'];

      // listOfSetValues = [];

      // for(let i=0;i<rowValues[0].length;i++)
      // {
      //   let colset = new Set();
      //   for(let j=0;j<rowValues.length;j++)
      //   {
      //     colset.add(rowValues[j][i]);
      //   }
      //   listOfSetValues.push(colset);
      // }

      // headers.forEach(colheader => {
      //   text += '<th scope="row">'+colheader;
      //   listOfSetValues.forEach(e => {
      //     arraySet = Array.from(e);
      //     text += arraySet+'</th>';
      //   });
      // });

'''