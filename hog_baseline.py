"""Train or evaluate the HOG / standardization / PCA / decision-tree baseline."""

import argparse
import pickle
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import MyCOCODataset
from data_split import split_annotation_ids, class_names
from mlmodel import Model
from settings import DATA_ROOT, ANNOTATIONS, OUTPUT_DIR, BATCH_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'evaluate'])
    args = parser.parse_args()
    coco = COCO(str(ANNOTATIONS))
    partitions = split_annotation_ids(coco)
    checkpoint = OUTPUT_DIR / 'hog_pca_tree.pkl'
    if args.mode == 'train':
        dataset = MyCOCODataset(DATA_ROOT, ANNOTATIONS, ids=partitions['train'], output_size=(128, 128))
        model = Model(hog_cell=16, hog_block=4, n_components=5)
        model.fit_pca(dataset, batch_size=BATCH_SIZE)
        model.fit_clf(dataset, batch_size=BATCH_SIZE)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with checkpoint.open('wb') as stream:
            pickle.dump({'model': model, 'category_ids': dataset.category_ids}, stream)
        print(f'Saved {checkpoint}')
        return

    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['category_ids'] != sorted(coco.cats):
        raise ValueError('Category IDs changed. Train the model with the current annotations.')
    model = saved['model']
    dataset = MyCOCODataset(DATA_ROOT, ANNOTATIONS, ids=partitions['test'], output_size=(128, 128))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=model._collate_fn)
    predicted, actual = [], []
    for images, labels in loader:
        predicted.append(model.forward(images))
        actual.append(labels)
    predicted = np.concatenate(predicted)
    actual = np.concatenate(actual)
    labels = list(range(len(coco.cats)))
    print('Confusion matrix (actual rows, predicted columns):')
    print(confusion_matrix(actual, predicted, labels=labels))
    print(classification_report(actual, predicted, labels=labels, target_names=class_names(coco), zero_division=0))


if __name__ == '__main__':
    main()
