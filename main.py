import warnings
import research

def load_test_data():
  list_of_instances = []
  with open('./data/test.csv') as input_stream:
    header_line = input_stream.readline()
    column_names = header_line[:-1].split(',')
    for line in input_stream:
      values = line[:-1].split(',')
      new_instance = dict(zip(column_names, values))
      list_of_instances.append(new_instance)

  return list_of_instances


def train_model():
  train_instances, train_labels = research.load_data()
  train_samples = map(research.to_sample, train_instances)
  model = research.get_model()
  model.fit(train_samples, train_labels)
  return model


if __name__ == '__main__':
  warnings.simplefilter('error')
  list_of_instances = load_test_data()
  # Use dump-load if training takes too much time or consistency is critical
  model = train_model()
  for instance in list_of_instances:
    sample = research.to_sample(instance)
    prediction = model.predict(sample)
    print(prediction)
    raw_input()

