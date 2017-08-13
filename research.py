import math
from sklearn.linear_model import LinearRegression

TRAIN_SAMPLES = 1000

def load_data():
  list_of_instances = []
  list_of_labels = []
  with open('./data/train.csv') as input_stream:
    header_line = input_stream.readline()
    column_names = header_line[:-1].split(',')
    for line in input_stream:
      values = line[:-1].split(',')
      new_instance = dict(zip(column_names[:-1], values[:-1]))
      new_label = values[-1]
      list_of_instances.append(new_instance)
      list_of_labels.append(int(new_label))

  return list_of_instances, list_of_labels


def get_lot_frontage(instance):
  if instance['LotFrontage'] == 'NA':
    return 0
  else:
    return int(instance['LotFrontage'])


def get_lot_area(instance):
  if instance['LotArea'] == 'NA':
    return 0
  else:
    return int(instance['LotArea'])


def to_sample(instance):
  return [get_lot_frontage(instance), get_lot_area(instance)]


def get_model():
  return LinearRegression()


if __name__ == '__main__':
  list_of_instances, list_of_labels = load_data()
  # print(list_of_instances[0])
  # print(list_of_labels[:5])
  list_of_samples = map(to_sample, list_of_instances)
  # print(list_of_samples[:5])
  model = LinearRegression()
  model.fit(list_of_samples[:TRAIN_SAMPLES], list_of_labels[:TRAIN_SAMPLES])

  predictions = []
  for sample, label in zip(list_of_samples[TRAIN_SAMPLES:], list_of_labels[TRAIN_SAMPLES:]):
    new_prediction = model.predict(sample)
    predictions.append(new_prediction)

  total_loss = 0
  for prediction, label in zip(predictions, list_of_labels[TRAIN_SAMPLES:]):
    new_loss = (math.log(prediction) - math.log(label)) ** 2
    total_loss += new_loss

  final_loss = math.sqrt(total_loss / len(predictions))

  print('Final score: {}'.format(final_loss))

