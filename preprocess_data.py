import tensorflow as tf
import os, cv2
import json, nltk
from collections import Counter
import csv

height = 512
width = 512
start_word = '<S>'
end_word = '</S>'
min_word_count = 4


class vocab(object):
    def __init__(self, vocab, unk_id):
        self.vocab = vocab
        self.unknown_id = unk_id

    def word_to_id(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else: return self.unknown_id


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _sequence_example(image, caption, vocab):
    image = cv2.resize(image, (width, height))
    image_raw = image.tostring()

    caption_ids = [vocab.word_to_id(word) for word in caption]
    caption_len = len(caption_ids)

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature={
            'caption_length':_int64_feature(caption_len),
            'height':_int64_feature(height),
            'width':_int64_feature(width),
            'image_raw':_bytes_feature(image_raw)
        }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'caption_ids':_int64_feature_list(caption_ids)
        })
    )

    return example


def create_vocab(captions):
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile('word_counts.txt', "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unknown_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

  vocabury = dict(enumerate(reverse_vocab))
  with open('vocabulary.csv', 'wb') as csv_file:
      writer = csv.writer(csv_file)
      for key, value in vocabury.items():
          writer.writerow([key, value])

  return vocab(vocab_dict, unknown_id)


def preprocess_caption(caption):
  tokenized_caption = [start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(end_word)
  return tokenized_caption


def write_data_to_tfrecords(data_dir, label_dir, writer):
    with tf.gfile.FastGFile(label_dir, 'r') as f:
        caption_data = json.load(f)

    id_to_filename = [(x['id'], x['file_name']) for x in caption_data['images']]
    id_to_captions = {}
    training_captions = []
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)
        training_captions.append(preprocess_caption(caption))

    print("Loaded caption metadata for %d images from %s" %
          (len(id_to_filename), label_dir))

    vocab = create_vocab(training_captions)
    cnt = 0

    for image_id, base_filename in id_to_filename:
        filename = os.path.join(data_dir, base_filename)
        image = cv2.imread(filename)
        captions = [preprocess_caption(caption) for caption in id_to_captions[image_id]]
        for caption in captions:
            sequence_example = _sequence_example(image, caption, vocab)
            writer.write(sequence_example.SerializeToString())
        cnt += 1
        if cnt == 20000:
            break

    return


data_dir = 'train2014'
label_dir = 'annotations/captions_train2014.json'
writer = tf.python_io.TFRecordWriter('train2014.tfrecords')
write_data_to_tfrecords(data_dir, label_dir, writer)
writer.close()