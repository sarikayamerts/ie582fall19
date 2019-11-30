import numpy as np
from sklearn.utils.validation import check_array

result_mapping = {1: [1, 0, 0],
                  0: [0, 1, 0],
                  2: [0, 0, 1]}

def ranked_probability_loss(obs, preds):
  """
  >>> y_true = [1, 1]
  >>> y_prob = [[0.5, 0.3, 0.2], [0.5, 0.2, 0.3]]
  >>> ranked_probability_loss(y_true, y_prob) # array([0.145, 0.17 ])

  >>> y_true = [1]
  >>> y_prob = [[0.7, 0.3, 0]]
  >>> ranked_probability_loss(y_true, y_prob) # array([0.045])
  """
  obs = check_array(obs, ensure_2d=False)
  preds = check_array(preds, ensure_2d=False)
  obs = np.array([result_mapping[i] for i in obs])

  cum_diff = np.cumsum(preds, axis=1) - np.cumsum(obs, axis=1)
  result = np.sum(np.square(cum_diff), axis=1)/2
  return np.round(result, 5)

def create_output(df):
  """
  Converts dataframe to comma separated string
  """
  output_list = df.to_string(header=False,
                             index=False,
                             index_names=False).split('\n')
  output_string = ','.join([','.join(ele.split()) for ele in output_list])
  return output_string