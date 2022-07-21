from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Node:
    name: str
    level: int
    children: List["Node"]
    parent: Optional["Node"]

    @classmethod
    def from_dict(cls, name, children: Dict, parent: "Node", level: int) -> "Node":
        node = cls(name=name, parent=parent, children=[], level=level)
        children = [
            cls.from_dict(name=name, parent=node, children=sub_children, level=level + 1)
            for name, sub_children in children.items()
        ]
        node.children = children
        return node

    def __repr__(self):
        text = f'{" - " * self.level}{self.name}\n'
        for node in self.children:
            text += str(node)
        return text

    def __iter__(self):
        for child in self.children:
            for node in child:
                yield node
        yield self


def fix_vector(top_level_node, label_to_index: Dict[str, int], labels: List[int]):
    for node in top_level_node:
        if node.name not in label_to_index:
            continue
        index = label_to_index[node.name]
        if labels[index] != 1:
            continue
        parent_node = node.parent
        if parent_node.name not in label_to_index:
            continue
        parent_index = label_to_index[parent_node.name]
        if labels[parent_index] != 1:
            labels[parent_index] = 1
    return labels


default_hierarchy = {
    "No Finding": {},
    "Support Devices": {},
    "Fracture": {},
    "Enlarged Cardiomediastinum": {"Cardiomegaly": {}},
    "Lung Opacity": {"Lung Lesion": {}, "Edema": {}, "Consolidation": {"Pneumonia": {}}, "Atelectasis": {}},
    "Pneumothorax": {},
    "Pleural Effusion": {},
    "Pleural Other": {},
}
default_hierarchy = Node.from_dict(None, children=default_hierarchy, parent=None, level=0)


if __name__ == "__main__":

    print(default_hierarchy)

    hierarchy = {
        "Enlarged Cardiomegaly": {"Cardiomegaly": {}},
        "Lung Opacity": {"Lung Lesion": {}, "Edema": {}, "Consolidation": {"Pneumonia": {}}, "Atelectasis": {}},
        "Pneumothorax": {},
    }

    index_to_label = [
        "Enlarged Cardiomegaly",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pneumothorax",
    ]
    label_to_index = {label: i for i, label in enumerate(index_to_label)}
    hierarchy = Node.from_dict(None, children=hierarchy, parent=None, level=0)

    labels = [1, 0, 0.5, 0, 0, 0.5, 1, 1]
    print(fix_vector(top_level_node=hierarchy, label_to_index=label_to_index, labels=labels))
