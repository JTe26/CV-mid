import xml.etree.ElementTree as ET
import os
import json

# Globals for COCO structure and counters
def init_coco():
    global coco, category_set, image_set, category_item_id, image_id, annotation_id
    coco = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}
    category_set = {}
    image_set = set()
    category_item_id = -1
    image_id = 20180000000
    annotation_id = 0

init_coco()

# Add a new category or retrieve existing
def addCatItem(name):
    global category_item_id
    if name in category_set:
        return category_set[name]
    category_item_id += 1
    category_item = {'supercategory': 'none', 'id': category_item_id, 'name': name}
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

# Add image metadata
def addImgItem(file_name, size):
    global image_id
    image_id += 1
    image_item = {'id': image_id, 'file_name': file_name,
                  'width': size['width'], 'height': size['height']}
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

# Add annotation for one object
def addAnnoItem(image_id, category_id, bbox):
    global annotation_id
    annotation_id += 1
    seg = [bbox[0], bbox[1], bbox[0], bbox[1] + bbox[3],
           bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]]
    anno = {
        'segmentation': [seg],
        'area': bbox[2] * bbox[3],
        'iscrowd': 0,
        'ignore': 0,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': annotation_id
    }
    coco['annotations'].append(anno)

# Read list of IDs from txt
def _read_image_ids(image_sets_file):
    with open(image_sets_file) as f:
        return [line.strip() for line in f]

# Parse one year/split without writing
def parse_voc_split(data_dir, split):
    image_sets_file = os.path.join(data_dir, 'ImageSets/Main', f'{split}.txt')
    if not os.path.exists(image_sets_file):
        print(f"Warning: No split file for {data_dir}/{split}, skipping.")
        return
    ids = _read_image_ids(image_sets_file)
    for _id in ids:
        xml_path = os.path.join(data_dir, 'Annotations', f'{_id}.xml')
        if not os.path.exists(xml_path):
            print(f"Warning: Annotation file missing {xml_path}, skipping object {_id}.")
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # read filename and size
        fname = root.findtext('filename')
        size_elem = root.find('size')
        size = {'width': int(size_elem.findtext('width')), 'height': int(size_elem.findtext('height'))}
        img_id = addImgItem(fname, size)
        # for each object in this annotation
        for obj in root.findall('object'):
            cat_name = obj.findtext('name')
            cat_id = addCatItem(cat_name)
            bbox_elem = obj.find('bndbox')
            bbox = [int(bbox_elem.findtext('xmin')),
                    int(bbox_elem.findtext('ymin')),
                    int(bbox_elem.findtext('xmax')) - int(bbox_elem.findtext('xmin')),
                    int(bbox_elem.findtext('ymax')) - int(bbox_elem.findtext('ymin'))]
            addAnnoItem(img_id, cat_id, bbox)

if __name__ == '__main__':
    base_dir = '/root/mmdetection/data/VOCdevkit'
    output_dir = '/root/mmdetection/data/combined_coco'
    os.makedirs(output_dir, exist_ok=True)

    years = ['VOC2007', 'VOC2012']
    splits = ['train', 'val', 'test']

    for split in splits:
        # reset globals and COCO container
        init_coco()
        # combine both years
        for year in years:
            parse_voc_split(os.path.join(base_dir, year), split)
        # write out combined JSON
        out_file = os.path.join(output_dir, f'voc0712_{split}.json')
        with open(out_file, 'w') as f:
            json.dump(coco, f)
        print(f'Saved {out_file}')
