from tqdm.auto import tqdm
from spacy.training import Example


def remove_from_list(l, idx):
    return l[:idx] + l[idx+1:]


def compute_scores(dataset, nlp):
    tp, fp, tn, fn = 0, 0, 0, 0
    total = 0
    for sample in tqdm(dataset):
        doc = nlp(sample[0])
        predicted_ = Example.from_dict(doc, dict(entities=extract_entities_span(doc))).to_dict()
        target_ = Example.from_dict(doc, sample[1]).to_dict()
        for target_ent, predicted_ent in zip(target_['doc_annotation']['entities'], predicted_['doc_annotation']['entities']):
            total += 1
            if predicted_ent!='O' and predicted_ent==target_ent:
                tp += 1 # True Positive
            elif predicted_ent!='O' and predicted_ent!=target_ent:
                fp += 1 # False Positive
            elif target_ent=='O' and predicted_ent==target_ent:
                tn += 1 # True Negative
            elif predicted_ent!=target_ent:
                fn += 1 # False Negative

    assert (tp + fp + fn + tn) == total

    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    f1 = 2.0 * (precision*recall) / (precision+recall)
    acc = (tp + tn) / total

    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def extract_entities_span(doc):
    # Get the spans of predicted entities from Spacy Doc object
    entities_span = []
    for ent in doc.ents:
        entities_span.append((ent.start_char, ent.end_char, ent.label_))
    return entities_span


def generate_batches(dataset, size, nlp):
    def prepare_examples(item):
        doc = nlp.make_doc(item[0])
        return Example.from_dict(doc, item[1])
    # Iterate over batches
    for i in range(0, len(dataset), size):
        batch = dataset[i: i+size]
        yield list(map(prepare_examples, batch))